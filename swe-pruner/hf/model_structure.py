import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import AutoConfig, AutoModel, AutoTokenizer


class CRFLayer(nn.Module):
    def __init__(self, num_tags: int = 2):
        """
        Args:
            num_tags: 2
            0=prune, 1=keep
        """
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor = None,
        mask: torch.Tensor = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        calc CRF-NLL

        Args:
            emissions: [B, L, num_tags]
            tags: [B, L]
            mask: [B, L]
            reduction: 'mean' | 'sum' | 'none'

        Returns:
            training: CRF-NLL
            decoding: best seq
        """
        if tags is not None:
            return self._compute_loss(emissions, tags, mask, reduction)
        else:
            # decoding mode
            return self._viterbi_decode(emissions, mask)

    def _compute_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        """calc CRF-NLL with sample-level norm"""
        # NLL = -log P(y|x) = log Z(x) - score(x, y)

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        gold_score = self._compute_score(emissions, tags, mask)
        forward_score = self._compute_normalizer(emissions, mask)
        nll = forward_score - gold_score

        # sample loss norm
        seq_lengths = mask.sum(dim=1).float().clamp(min=1)  # [B]
        nll = nll / seq_lengths

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            valid = mask[:, i]
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, i], tags[:, i - 1]]
            score = score + (emit_score + trans_score) * valid

        last_tags = tags.gather(1, mask.sum(dim=1).long().unsqueeze(1) - 1).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        # alpha[b, t] = log(sum over all paths ending at tag t at position 0)
        alpha = self.start_transitions + emissions[:, 0]  # [B, num_tags]

        for i in range(1, seq_len):
            # [B, num_tags]
            emit_score = emissions[:, i]

            # [num_tags, num_tags]
            trans_score = self.transitions

            # alpha_new[j] = log sum_i exp(alpha[i] + trans[j,i] + emit[j])
            # log-sum-exp trick
            alpha_expanded = alpha.unsqueeze(1)  # [B, 1, num_tags]
            trans_expanded = trans_score.unsqueeze(0)  # [1, num_tags, num_tags]

            # [B, num_tags, num_tags]
            scores = alpha_expanded + trans_expanded  # [B, num_tags, num_tags]
            alpha_new = torch.logsumexp(scores, dim=2) + emit_score  # [B, num_tags]

            alpha = torch.where(mask[:, i].unsqueeze(1), alpha_new, alpha)

        alpha += self.end_transitions

        return torch.logsumexp(alpha, dim=1)  # [B]

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape

        if mask is None:
            mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=emissions.device
            )

        viterbi_score = self.start_transitions + emissions[:, 0]  # [B, num_tags]
        backpointers = []

        for i in range(1, seq_len):
            emit_score = emissions[:, i]

            # [B, num_tags, 1] + [1, num_tags, num_tags] = [B, num_tags, num_tags]
            scores = viterbi_score.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_scores, best_tags = scores.max(dim=1)  # [B, num_tags]

            viterbi_score_new = max_scores + emit_score
            viterbi_score = torch.where(
                mask[:, i].unsqueeze(1), viterbi_score_new, viterbi_score
            )
            backpointers.append(best_tags)

        viterbi_score += self.end_transitions
        best_last_tags = viterbi_score.argmax(dim=1)  # [B]

        best_path = [best_last_tags]
        for bp in reversed(backpointers):
            best_path.append(bp.gather(1, best_path[-1].unsqueeze(1)).squeeze(1))

        best_path = torch.stack(best_path[::-1], dim=1)  # [B, L]

        return best_path


class CRFCompressionHead(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 2, dtype=torch.float32),
        )

        self.crf = CRFLayer(num_tags=2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        emissions = self.feature_extractor(x)  # [B, L, 2]
        return emissions

    def compute_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        return self.crf(emissions, tags, mask, reduction)

    def decode(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        emissions = self.feature_extractor(x)
        return self.crf._viterbi_decode(emissions, mask)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        emissions = self.feature_extractor(x)  # [B, L, 2]
        probs = F.softmax(emissions, dim=-1)[:, :, 1]  # [B, L]
        return probs


class TokenScorer(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer = None,
        bottleneck: int = 256,
        dropout: float = 0.1,
        num_fusion_layers: int = 1,
        num_heads: int = 8,
        use_multi_layer_fusion: bool = False,
        early_layer_ratio: float = 0.25,
        middle_layer_ratio: float = 0.5,
        compression_head_type: str = "ffn",
        *,
        load_pretrained_backbone: bool = True,
        backbone_config: Optional[AutoConfig] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = "flash_attention_2",
        # HINT: requires flash attn 2
    ):
        super().__init__()
        self.use_multi_layer_fusion = use_multi_layer_fusion

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, None)

        if load_pretrained_backbone:
            backbone_kwargs = {
                "device_map": None,
                "trust_remote_code": trust_remote_code,
            }
            if torch_dtype is not None:
                backbone_kwargs["torch_dtype"] = torch_dtype
            if attn_implementation is not None:
                backbone_kwargs["attn_implementation"] = attn_implementation
            self.backbone = AutoModel.from_pretrained(model_name, **backbone_kwargs)
        else:
            if backbone_config is None:
                backbone_config = AutoConfig.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
            if torch_dtype is not None:
                backbone_config.torch_dtype = torch_dtype
            self.backbone = AutoModel.from_config(
                backbone_config, trust_remote_code=trust_remote_code
            )
            if attn_implementation is not None and hasattr(
                self.backbone, "set_attn_implementation"
            ):
                try:
                    self.backbone.set_attn_implementation(attn_implementation)
                except Exception:
                    pass
            if torch_dtype is not None:
                self.backbone.to(dtype=torch_dtype)
        hidden_size = self.backbone.config.hidden_size

        if self.use_multi_layer_fusion:
            num_layers = self.backbone.config.num_hidden_layers
            self.early_layer_idx = max(1, int(num_layers * early_layer_ratio))
            self.middle_layer_idx = max(1, int(num_layers * middle_layer_ratio))
            self.final_layer_idx = num_layers

            self.fused_hidden_size = hidden_size * 3

        else:
            self.fused_hidden_size = hidden_size

        embedding_layer = self.backbone.get_input_embeddings()
        if embedding_layer is None:
            raise ValueError("Backbone does not expose input embeddings")
        self.embedding_layer = embedding_layer

        if tokenizer:
            self.token_yes_id = tokenizer.convert_tokens_to_ids("yes")
            self.token_no_id = tokenizer.convert_tokens_to_ids("no")
            # Determine if this is an LLM-style model (for score calculation)
            # LLM-style models don't have cls_token_id or sep_token_id
            self.is_llm = (
                tokenizer.cls_token_id is None or tokenizer.sep_token_id is None
            )
        else:
            self.is_llm = True  # Default to True for LLM-style
        self.backbone.eval()

        self.dropout = nn.Dropout(dropout)
        self.num_fusion_layers = num_fusion_layers
        self.num_heads = num_heads
        self.fusion_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.fused_hidden_size,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_fusion_layers)
            ]
        )
        self.fusion_norms = nn.ModuleList(
            [nn.LayerNorm(self.fused_hidden_size) for _ in range(num_fusion_layers)]
        )

        self.compression_head_type = compression_head_type
        if compression_head_type == "ffn":
            # Transformer-style FFN: LayerNorm -> expand -> GELU -> project
            expansion_dim = bottleneck * 2
            self.compression_head = nn.Sequential(
                nn.LayerNorm(self.fused_hidden_size),
                nn.Linear(self.fused_hidden_size, expansion_dim, dtype=torch.float32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expansion_dim, bottleneck, dtype=torch.float32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, 1, dtype=torch.float32),
            )
        elif compression_head_type == "simple":
            self.compression_head = nn.Sequential(
                nn.Linear(self.fused_hidden_size, bottleneck, dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(bottleneck, 1, dtype=torch.float32),
            )
        elif compression_head_type == "crf":
            self.compression_head = CRFCompressionHead(
                self.fused_hidden_size, bottleneck, dropout
            )
        else:
            raise ValueError(f"Unknown compression_head_type: {compression_head_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.use_multi_layer_fusion:
            all_hidden_states = backbone_outputs.hidden_states

            early_hidden = all_hidden_states[self.early_layer_idx].float()  # [B, L, H]
            middle_hidden = all_hidden_states[
                self.middle_layer_idx
            ].float()  # [B, L, H]
            final_hidden = all_hidden_states[self.final_layer_idx].float()  # [B, L, H]
            fused_hidden = torch.cat(
                [early_hidden, middle_hidden, final_hidden], dim=-1
            )

            h_for_compression = fused_hidden
        else:
            raw_last_hidden = backbone_outputs.hidden_states[-1].float()  # [B, L, H]
            h_for_compression = raw_last_hidden

        last_hidden = backbone_outputs.last_hidden_state.float()  # [B, L, H]
        h_for_scoring = last_hidden

        h = h_for_compression
        key_padding_mask = (attention_mask == 0).to(h.device)

        for attn_layer, norm_layer in zip(self.fusion_layers, self.fusion_norms):
            attn_output, attn_weights = attn_layer(
                h,
                h,
                h,
                key_padding_mask=key_padding_mask,
            )
            h = norm_layer(attn_output + h)
        h_compression = self.dropout(h)

        if self.compression_head_type == "crf":
            token_emissions = self.compression_head(h_compression)  # [B, L, 2]
            token_logits = token_emissions[:, :, 1] - token_emissions[:, :, 0]  # [B, L]
        else:
            token_logits = self.compression_head(h_compression).squeeze(-1)  # [B, L]

        batch_size = h_for_scoring.size(0)
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_indices = torch.clamp(last_token_indices, min=0)

        last_hidden_for_scoring = h_for_scoring[
            torch.arange(batch_size), last_token_indices
        ]

        embedding_weight = self.embedding_layer.weight.float()

        last_token_logits = torch.matmul(
            last_hidden_for_scoring, embedding_weight.T
        )  # [B, vocab_size]

        no_vector = last_token_logits[:, self.token_no_id]
        yes_vector = last_token_logits[:, self.token_yes_id]

        logits_stack = torch.stack([no_vector, yes_vector], dim=1)
        log_probs = F.log_softmax(logits_stack, dim=1)
        score_logits = log_probs[:, 1]

        result = {
            "token_logits": token_logits,
            "score_logits": score_logits,
        }
        return result
