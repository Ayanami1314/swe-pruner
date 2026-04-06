import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pydantic import BaseModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.tensorboard import SummaryWriter
import typer
from rich.console import Console
from tqdm import tqdm
import torchmetrics
import os

console = Console()


def setup_ddp():
    """Initialize DDP environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if current process is main process"""
    return rank == 0


class ResidualBlock(nn.Module):
    """单个残差块，用于深层MLP"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim, dtype=torch.float32),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))  # 残差连接


class ResidualMLP(nn.Module):
    """带残差连接的多层MLP，用于token分类"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)

        # 两个残差块
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(2)]
        )

        self.output_proj = nn.Linear(hidden_dim, 1, dtype=torch.float32)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [B, L, input_dim]
        x = self.norm(x)
        x = self.input_proj(x)  # [B, L, hidden_dim]

        for block in self.blocks:
            x = block(x)

        return self.output_proj(x)  # [B, L, 1]


class CRFLayer(nn.Module):
    """
    条件随机场(CRF)层，用于序列标注任务。

    CRF的优势：
    1. 考虑标签之间的依赖关系（转移概率）
    2. 全局最优解码（Viterbi算法）
    3. 避免不合理的标签序列

    对于二分类任务(0=prune, 1=keep)，CRF可以学习到：
    - 连续的"保留"区域更合理
    - 避免频繁的0-1交替
    """

    def __init__(self, num_tags: int = 2):
        """
        Args:
            num_tags: 标签数量，默认2（二分类：0=prune, 1=keep）
        """
        super().__init__()
        self.num_tags = num_tags

        # 转移矩阵: transitions[i, j] = 从标签j转移到标签i的分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # 起始和结束的转移分数
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        # 初始化转移矩阵（鼓励连续的相同标签）
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor = None,
        mask: torch.Tensor = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        计算CRF的负对数似然损失（用于训练）或最优标签序列（用于解码）

        Args:
            emissions: [B, L, num_tags] 发射分数（来自上游网络）
            tags: [B, L] 真实标签序列（训练时提供）
            mask: [B, L] 有效位置的mask（True表示有效）
            reduction: 'mean' | 'sum' | 'none'

        Returns:
            训练时: 负对数似然损失
            解码时: 最优标签序列
        """
        if tags is not None:
            # 训练模式：计算负对数似然
            return self._compute_loss(emissions, tags, mask, reduction)
        else:
            # 解码模式：Viterbi算法找最优路径
            return self._viterbi_decode(emissions, mask)

    def _compute_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        """计算CRF的负对数似然损失（归一化到每个token）"""
        # NLL = -log P(y|x) = log Z(x) - score(x, y)
        # 其中 Z(x) 是配分函数，score(x, y) 是真实路径的分数

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        # 计算真实路径分数
        gold_score = self._compute_score(emissions, tags, mask)

        # 计算配分函数（所有可能路径的分数之和的log）
        forward_score = self._compute_normalizer(emissions, mask)

        # 负对数似然
        nll = forward_score - gold_score

        # 【关键修改】按序列长度归一化，使loss与BCE/Focal Loss在数值上可比
        # 这样避免长序列产生过大的loss值
        seq_lengths = mask.sum(dim=1).float().clamp(min=1)  # [B]
        nll = nll / seq_lengths  # 每个token的平均NLL

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """计算给定标签序列的分数"""
        batch_size, seq_len = tags.shape

        # 起始分数
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            # 只在有效位置计算
            valid = mask[:, i]

            # 发射分数
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)

            # 转移分数 (从 tags[:, i-1] 转移到 tags[:, i])
            trans_score = self.transitions[tags[:, i], tags[:, i - 1]]

            # 累加分数（仅对有效位置）
            score = score + (emit_score + trans_score) * valid

        # 结束分数
        # 找到每个序列的最后一个有效位置
        last_tags = tags.gather(1, mask.sum(dim=1).long().unsqueeze(1) - 1).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """使用前向算法计算配分函数（log-space）"""
        batch_size, seq_len, _ = emissions.shape

        # 初始化：起始分数 + 第一个位置的发射分数
        # alpha[b, t] = log(sum over all paths ending at tag t at position 0)
        alpha = self.start_transitions + emissions[:, 0]  # [B, num_tags]

        for i in range(1, seq_len):
            # 发射分数: [B, num_tags]
            emit_score = emissions[:, i]

            # 转移分数: [num_tags, num_tags]
            # transitions[j, i] = 从i转移到j的分数
            trans_score = self.transitions

            # 计算新的alpha
            # alpha_new[j] = log sum_i exp(alpha[i] + trans[j,i] + emit[j])
            # 使用log-sum-exp trick避免数值溢出
            alpha_expanded = alpha.unsqueeze(1)  # [B, 1, num_tags]
            trans_expanded = trans_score.unsqueeze(0)  # [1, num_tags, num_tags]

            # [B, num_tags, num_tags]: 从每个前一状态到每个当前状态的分数
            scores = alpha_expanded + trans_expanded  # [B, num_tags, num_tags]
            alpha_new = torch.logsumexp(scores, dim=2) + emit_score  # [B, num_tags]

            # 只在有效位置更新
            alpha = torch.where(mask[:, i].unsqueeze(1), alpha_new, alpha)

        # 加上结束转移分数
        alpha += self.end_transitions

        # 最终配分函数
        return torch.logsumexp(alpha, dim=1)  # [B]

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Viterbi解码：找到最优标签序列"""
        batch_size, seq_len, _ = emissions.shape

        if mask is None:
            mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=emissions.device
            )

        # 初始化
        viterbi_score = self.start_transitions + emissions[:, 0]  # [B, num_tags]
        backpointers = []

        for i in range(1, seq_len):
            emit_score = emissions[:, i]

            # 计算从所有前一状态转移到当前状态的分数
            # [B, num_tags, 1] + [1, num_tags, num_tags] = [B, num_tags, num_tags]
            scores = viterbi_score.unsqueeze(2) + self.transitions.unsqueeze(0)

            # 找最大值和对应的前一状态
            max_scores, best_tags = scores.max(dim=1)  # [B, num_tags]

            viterbi_score_new = max_scores + emit_score
            viterbi_score = torch.where(
                mask[:, i].unsqueeze(1), viterbi_score_new, viterbi_score
            )
            backpointers.append(best_tags)

        # 加上结束转移分数，找最优的最后一个标签
        viterbi_score += self.end_transitions
        best_last_tags = viterbi_score.argmax(dim=1)  # [B]

        # 回溯
        best_path = [best_last_tags]
        for bp in reversed(backpointers):
            best_path.append(bp.gather(1, best_path[-1].unsqueeze(1)).squeeze(1))

        # 反转得到正确顺序
        best_path = torch.stack(best_path[::-1], dim=1)  # [B, L]

        return best_path


class CRFCompressionHead(nn.Module):
    """
    结合MLP特征提取和CRF序列建模的压缩头

    结构：
    1. MLP: 将hidden states映射到2维发射分数
    2. CRF: 基于发射分数进行序列标注
    """

    def __init__(self, input_dim: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()

        # 特征提取MLP
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 2, dtype=torch.float32),  # 2类：prune/keep
        )

        # CRF层
        self.crf = CRFLayer(num_tags=2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播，返回发射分数（用于训练时计算loss）

        Args:
            x: [B, L, input_dim] hidden states
            mask: [B, L] 有效位置mask

        Returns:
            emissions: [B, L, 2] 发射分数
        """
        emissions = self.feature_extractor(x)  # [B, L, 2]
        return emissions

    def compute_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算CRF损失"""
        return self.crf(emissions, tags, mask, reduction)

    def decode(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """解码：返回最优标签序列"""
        emissions = self.feature_extractor(x)
        return self.crf._viterbi_decode(emissions, mask)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取每个位置为正类（keep）的概率
        用于与MLP head的输出保持一致的接口

        Returns:
            probs: [B, L] 正类概率
        """
        emissions = self.feature_extractor(x)  # [B, L, 2]
        # 使用softmax将发射分数转为概率
        probs = F.softmax(emissions, dim=-1)[:, :, 1]  # [B, L]
        return probs


class TokenScorer(nn.Module):
    """
    双头模型：body + 打分头 & 压缩头
    - 压缩头：token-level MLP，输出 [B, L] logits，用于压缩损失
    - 打分头：LLM使用共享的input embedding权重作为lm_head
    - 多层特征融合：拼接前层、中层、后层的hidden states增强语义表达
    """

    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer = None,
        bottleneck: int = 256,
        dropout: float = 0.1,
        num_finetune_layers: int = 0,
        num_fusion_layers: int = 1,
        num_heads: int = 8,
        use_multi_layer_fusion: bool = False,
        early_layer_ratio: float = 0.25,
        middle_layer_ratio: float = 0.5,
        compression_head_type: str = "ffn",
    ):
        super().__init__()
        self.is_llm = True  # LLM-only (BERT path removed)
        self.use_multi_layer_fusion = use_multi_layer_fusion

        # 1. 统一使用AutoModel加载backbone
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map=None,
        )
        hidden_size = self.backbone.config.hidden_size

        # 2. 计算多层特征融合的层索引
        if self.use_multi_layer_fusion:
            num_layers = self.backbone.config.num_hidden_layers
            self.early_layer_idx = max(1, int(num_layers * early_layer_ratio))
            self.middle_layer_idx = max(1, int(num_layers * middle_layer_ratio))
            self.final_layer_idx = num_layers  # 最后一层

            # 融合后的特征维度是3倍
            self.fused_hidden_size = hidden_size * 3

            if is_main_process(0):
                console.print(
                    f"Multi-layer fusion enabled: layers {self.early_layer_idx}, "
                    f"{self.middle_layer_idx}, {self.final_layer_idx} (total {num_layers} layers)"
                )
                console.print(f"Fused feature dimension: {self.fused_hidden_size}")
        else:
            self.fused_hidden_size = hidden_size

        # 3. Word embedding as lm_head for yes/no scoring
        self.word_embeddings = self.backbone.get_input_embeddings().weight
        if tokenizer:
            self.token_yes_id = tokenizer.convert_tokens_to_ids("yes")
            self.token_no_id = tokenizer.convert_tokens_to_ids("no")

        # 4. 冻结/解冻backbone参数的逻辑保持不变
        for p in self.backbone.parameters():
            p.requires_grad = False

        if num_finetune_layers > 0:
            # ... (您的解冻逻辑保持不变) ...
            if hasattr(self.backbone, "transformer") and hasattr(
                self.backbone.transformer, "h"
            ):
                layers = self.backbone.transformer.h
                num_layers = len(layers)
                layers_to_finetune = num_layers - num_finetune_layers
                for idx in range(layers_to_finetune, num_layers):
                    for p in layers[idx].parameters():
                        p.requires_grad = True
            # ... (fallback逻辑) ...

        self.backbone.eval()

        # 5. 压缩头和融合层 - 使用融合后的特征维度
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

        # 【改进】使用现代化的压缩头架构
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
            # 保留原始简单结构（向后兼容）
            self.compression_head = nn.Sequential(
                nn.Linear(self.fused_hidden_size, bottleneck, dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(bottleneck, 1, dtype=torch.float32),
            )
        elif compression_head_type == "residual":
            # 带残差连接的深层MLP
            self.compression_head = ResidualMLP(
                self.fused_hidden_size, bottleneck, dropout
            )
        elif compression_head_type == "crf":
            # CRF序列建模压缩头
            self.compression_head = CRFCompressionHead(
                self.fused_hidden_size, bottleneck, dropout
            )
        else:
            raise ValueError(f"Unknown compression_head_type: {compression_head_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # 前向获取hidden states
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 必须开启以获取所有层的hidden states
            return_dict=True,
        )

        # 【多层特征融合】提取前、中、后三层的hidden states并拼接
        if self.use_multi_layer_fusion:
            # hidden_states是一个tuple: (embedding_output, layer_1, ..., layer_n)
            # 索引0是embedding，索引1-n是各transformer层的输出
            all_hidden_states = backbone_outputs.hidden_states

            early_hidden = all_hidden_states[self.early_layer_idx].float()  # [B, L, H]
            middle_hidden = all_hidden_states[
                self.middle_layer_idx
            ].float()  # [B, L, H]
            final_hidden = all_hidden_states[self.final_layer_idx].float()  # [B, L, H]

            # 拼接三层特征 -> [B, L, 3H]
            fused_hidden = torch.cat(
                [early_hidden, middle_hidden, final_hidden], dim=-1
            )

            h_for_compression = fused_hidden  # 压缩头使用融合特征
        else:
            # 原始逻辑：只使用最后一层
            raw_last_hidden = backbone_outputs.hidden_states[-1].float()  # [B, L, H]
            h_for_compression = raw_last_hidden

        # 打分头始终使用经过LayerNorm的last_hidden_state
        last_hidden = backbone_outputs.last_hidden_state.float()  # [B, L, H]
        h_for_scoring = last_hidden

        # --- 压缩头逻辑 ---
        h = h_for_compression
        key_padding_mask = (attention_mask == 0).to(h.device)
        attention_weights_list = []

        for attn_layer, norm_layer in zip(self.fusion_layers, self.fusion_norms):
            attn_output, attn_weights = attn_layer(
                h,
                h,
                h,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention,
            )
            if return_attention:
                attention_weights_list.append(attn_weights)
            h = norm_layer(attn_output + h)
        h_compression = self.dropout(h)

        # 根据压缩头类型处理输出
        if self.compression_head_type == "crf":
            # CRF头返回发射分数 [B, L, 2]，需要特殊处理
            token_emissions = self.compression_head(h_compression)  # [B, L, 2]
            # 为了与其他head保持一致的接口，将class 1的logit作为token_logits
            token_logits = token_emissions[:, :, 1] - token_emissions[:, :, 0]  # [B, L]
        else:
            token_logits = self.compression_head(h_compression).squeeze(-1)  # [B, L]

        # --- Scoring head (LLM: yes/no logprob from last token) ---
        batch_size = h_for_scoring.size(0)
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_indices = torch.clamp(last_token_indices, min=0)
        last_hidden_for_scoring = h_for_scoring[
            torch.arange(batch_size), last_token_indices
        ]
        word_embeddings_float32 = self.word_embeddings.float()
        last_token_logits = torch.matmul(
            last_hidden_for_scoring, word_embeddings_float32.T
        )
        no_vector = last_token_logits[:, self.token_no_id]
        yes_vector = last_token_logits[:, self.token_yes_id]
        logits_stack = torch.stack([no_vector, yes_vector], dim=1)
        log_probs = F.log_softmax(logits_stack, dim=1)
        score_logits = log_probs[:, 1]

        result = {
            "token_logits": token_logits,
            "score_logits": score_logits,
        }

        if return_attention:
            result["attention_weights"] = attention_weights_list
            if self.use_multi_layer_fusion:
                result["early_hidden"] = early_hidden
                result["middle_hidden"] = middle_hidden
                result["final_hidden"] = final_hidden

        return result


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    用于处理类别不平衡，gamma越大越关注困难样本

    Args:
        alpha: 平衡因子，用于处理类别不平衡 (default: 0.25), alpha代表正样本的权重
        gamma: 聚焦参数，gamma越大越关注困难样本 (default: 2.0)
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N] 原始logits
            targets: [N] 0或1的标签
        """
        # 计算概率
        probs = torch.sigmoid(logits)

        # 计算p_t: 对于正样本用p，负样本用1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # 计算alpha_t: 对于正样本用alpha，负样本用1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DictData(BaseModel):
    query: str
    code: str
    kept_frags: List[int]
    score: float

    class Config:
        extra = "allow"


def format_instruction(instruction: str, query: str) -> str:
    """格式化instruction和query部分（LLM风格）"""
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "


def kept_frags_to_label(
    kept_frags: List[int],
    code: str,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    """
    kept_frags : 1-based line numbers to KEEP (mask=1), others mask=0
    code       : 原始代码字符串
    return     : 1-D torch.FloatTensor, length = len(tokenize(code)), 1.0=keep, 0.0=prune
    """
    # 1. 计算每行的字符区间
    lines = code.splitlines(keepends=True)
    keep_char_spans = []
    char_cnt = 0
    for idx, line in enumerate(lines, 1):
        if idx in kept_frags:
            keep_char_spans.append((char_cnt, char_cnt + len(line)))
        char_cnt += len(line)

    # 2. tokenize code（不加特殊token，因为会在pair encoding时统一处理）
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    code_tokens = enc["input_ids"]
    offsets = enc["offset_mapping"]  # List[(start, end)]

    # 3. 构造 mask：遍历每个code token，检查其字符区间是否与kept行重叠
    mask = torch.zeros(len(code_tokens), dtype=torch.float32)
    for i, (tok_s, tok_e) in enumerate(offsets):
        # 只要token与任意保留行有交集，就标记为1
        for ks, ke in keep_char_spans:
            if tok_s < ke and tok_e > ks:  # 有交集
                mask[i] = 1.0
                break

    return mask


class CodePruneDataset(Dataset):
    def __init__(
        self,
        data: List[DictData],
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
        instruction: str = None,
        compute_class_ratio: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction

        self.pos_ratio = None
        self.neg_ratio = None
        self.auto_focal_alpha = None

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = tokenizer.encode(self.suffix, add_special_tokens=False)

        # 计算正负类比例并自动设置focal alpha
        if compute_class_ratio:
            self._compute_class_statistics()

    def _compute_class_statistics(self):
        """
        统计数据集中正负类的比例，并自动计算focal loss的alpha参数。

        Focal Loss中的alpha参数用于平衡正负类：
        - alpha越大，对正类的惩罚越大
        - 通常设置为 1 - pos_ratio，即少数类获得更大的权重

        这里采用的策略是：alpha = 1 - pos_ratio，使得少数类（正类）获得更大的权重
        """
        total_pos_tokens = 0
        total_tokens = 0

        console.print("Computing class statistics from dataset...")

        # 采样计算（如果数据集太大，只采样一部分）
        sample_size = min(len(self.data), 1000)  # 最多采样1000条数据
        sample_indices = list(
            range(0, len(self.data), max(1, len(self.data) // sample_size))
        )[:sample_size]

        for idx in tqdm(
            sample_indices,
            desc="Computing class ratio",
            disable=len(sample_indices) < 100,
        ):
            item = self.data[idx]
            # 计算这条数据的code token级别的label
            code_mask = kept_frags_to_label(
                item.kept_frags,
                code=item.code,
                tokenizer=self.tokenizer,
            )
            total_pos_tokens += code_mask.sum().item()
            total_tokens += code_mask.numel()

        if total_tokens > 0:
            self.pos_ratio = total_pos_tokens / total_tokens
            self.neg_ratio = 1 - self.pos_ratio
            # alpha设置为正类的比例（这样负类会获得1-alpha的权重，负类多时权重小）
            # 或者使用1 - pos_ratio让少数类获得更大权重
            self.auto_focal_alpha = self.pos_ratio  # 少数类（正类）获得更大权重

            console.print(
                f"Class statistics computed from {len(sample_indices)} samples:"
            )
            console.print(
                f"  - Positive token ratio: {self.pos_ratio:.4f} ({self.pos_ratio * 100:.2f}%)"
            )
            console.print(
                f"  - Negative token ratio: {self.neg_ratio:.4f} ({self.neg_ratio * 100:.2f}%)"
            )
            console.print(
                f"  - Recommended focal_alpha (auto): {self.auto_focal_alpha:.4f}"
            )
        else:
            console.print("[yellow]No valid tokens found for class statistics[/yellow]")
            self.pos_ratio = 0.5
            self.neg_ratio = 0.5
            self.auto_focal_alpha = 0.5

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._getitem_llm(self.data[idx])

    def _getitem_llm(self, item: DictData) -> Dict[str, Any]:
        """LLM风格：prefix + format_instruction(query) + code + suffix"""
        # 1. 格式化query部分（包含instruction和query，但不包含code）
        formatted_query = format_instruction(self.instruction, item.query)

        # 2. tokenize格式化后的query和code（都不加特殊token）
        query_enc = self.tokenizer(
            formatted_query,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        code_enc = self.tokenizer(
            item.code,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )

        query_ids = query_enc["input_ids"]
        code_ids = code_enc["input_ids"]

        # 3. 计算可用长度：total - prefix - suffix
        available_length = (
            self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        query_len = len(query_ids)
        code_len = len(code_ids)

        # 4. 如果超限，只截断code
        if query_len + code_len > available_length:
            code_ids = code_ids[: available_length - query_len]
            code_len = len(code_ids)

        # 5. 拼接：prefix + query + code + suffix
        input_ids = self.prefix_tokens + query_ids + code_ids + self.suffix_tokens
        real_len = len(input_ids)

        # 6. 【关键修改】RIGHT padding for LLM (与官方完全一致)
        pad_len = self.max_length - real_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len  # Padding在右边
        attention_mask = [1] * real_len + [0] * pad_len  # Mask也在右边

        # 7. 【关键修改】doc_mask 的计算也需要相应调整
        # RIGHT padding后，code在序列中的位置：prefix + query + **code** + suffix + [pad...]
        doc_start = len(self.prefix_tokens) + query_len
        doc_end = doc_start + code_len

        doc_mask = torch.zeros(self.max_length, dtype=torch.bool)
        doc_mask[doc_start:doc_end] = True

        # 8. code_labels 的逻辑保持不变，因为doc_start/doc_end是正确的
        code_mask = kept_frags_to_label(
            item.kept_frags,
            code=item.code,
            tokenizer=self.tokenizer,
        )
        code_mask = code_mask[:code_len]
        token_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        token_labels[doc_start:doc_end] = code_mask.long()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "doc_mask": doc_mask,
            "token_labels": token_labels,
            "score": torch.tensor(item.score, dtype=torch.float32),
        }


def compute_combined_loss(
    model: TokenScorer,
    batch: Dict[str, Any],
    lambda_score: float = 0.05,
    device: torch.device = None,
    compression_loss_type: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    use_sample_level_aggregation: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算组合损失：压缩损失 * (1 - lambda) + 得分损失 * lambda
      - 压缩损失：支持BCE、Focal Loss或CRF，仅在 doc_mask & attention_mask 的位置计算
      - 得分损失：MSE between predicted scores and ground truth scores

    Args:
        compression_loss_type: 'bce', 'focal', or 'crf'
        focal_alpha: Focal Loss的alpha参数（仅当loss_type='focal'时使用）
        focal_gamma: Focal Loss的gamma参数（仅当loss_type='focal'时使用）
        use_sample_level_aggregation: 是否使用样本层面的损失聚合，避免长样本主导梯度更新
            - True: 先对每个样本内的token loss取平均，再对batch内的样本取平均
            - False: 对所有valid token直接取全局平均（原始行为）

    返回：loss 以及日志
    """
    if device is None:
        device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    doc_mask = batch["doc_mask"].to(device).bool()
    token_labels = batch["token_labels"].to(device)
    ground_truth_scores = batch["scores"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    token_logits = outputs["token_logits"]  # [B, L]
    score_logits = outputs["score_logits"]  # [B]

    # 处理DDP包装的模型
    actual_model = model.module if hasattr(model, "module") else model

    # 计算压缩损失
    valid_mask = doc_mask & attention_mask.bool() & (token_labels != -100)
    compress_loss = torch.tensor(0.0, device=device)
    pos_rate = 0.0

    if valid_mask.sum() > 0:
        # 检查是否使用CRF头
        use_crf = actual_model.compression_head_type == "crf"

        if use_crf:
            # CRF损失计算：需要特殊处理
            # 对于CRF，我们需要在doc_mask范围内计算序列级别的损失
            batch_size = token_logits.size(0)
            sample_losses = []
            total_pos_tokens = 0
            total_valid_tokens = 0

            for i in range(batch_size):
                sample_valid_mask = valid_mask[i]  # [L]
                if sample_valid_mask.sum() == 0:
                    continue

                # 找到doc区域的起始和结束位置
                valid_positions = sample_valid_mask.nonzero(as_tuple=True)[0]
                start_pos = valid_positions[0].item()
                end_pos = valid_positions[-1].item() + 1

                # 提取doc区域的hidden states和labels
                # 需要重新通过compression_head获取emissions
                # 但这里我们可以从token_logits反推或者直接使用模型
                sample_labels = token_labels[i, start_pos:end_pos]  # [doc_len]

                # 统计正负类信息
                total_pos_tokens += (sample_labels == 1).sum().item()
                total_valid_tokens += sample_labels.numel()

                # 获取该样本的发射分数
                # 由于CRF头的forward已经计算了emissions，我们需要重新获取
                # 这里使用一个简化的方法：直接用token_logits作为二分类的差值
                # emissions = [neg_score, pos_score], token_logits = pos_score - neg_score
                # 重建emissions: neg = -logits/2, pos = logits/2
                sample_logits = token_logits[i, start_pos:end_pos]  # [doc_len]

                # 重建发射分数：使用对称方式
                emissions_sample = torch.stack(
                    [
                        -sample_logits / 2,  # class 0 (prune)
                        sample_logits / 2,  # class 1 (keep)
                    ],
                    dim=-1,
                ).unsqueeze(0)  # [1, doc_len, 2]

                # 创建该样本的mask
                sample_crf_mask = torch.ones(
                    1, end_pos - start_pos, dtype=torch.bool, device=device
                )

                # 计算CRF损失
                sample_loss = actual_model.compression_head.crf(
                    emissions_sample,
                    sample_labels.unsqueeze(0),  # [1, doc_len]
                    sample_crf_mask,
                    reduction="mean",
                )
                sample_losses.append(sample_loss)

            if len(sample_losses) > 0:
                compress_loss = torch.stack(sample_losses).mean()
                pos_rate = total_pos_tokens / max(total_valid_tokens, 1)
        elif use_sample_level_aggregation:
            # 【样本层面聚合】先对每个样本内的token loss取平均，再对batch内的样本取平均
            # 这样避免了长样本因token数量多而主导梯度更新
            batch_size = token_logits.size(0)
            sample_losses = []
            total_pos_tokens = 0
            total_valid_tokens = 0

            for i in range(batch_size):
                sample_valid_mask = valid_mask[i]  # [L]
                if sample_valid_mask.sum() == 0:
                    continue

                sample_logits = token_logits[i][sample_valid_mask]  # [num_valid_tokens]
                sample_labels = token_labels[i][
                    sample_valid_mask
                ].float()  # [num_valid_tokens]

                # 统计正负类信息
                total_pos_tokens += sample_labels.sum().item()
                total_valid_tokens += sample_labels.numel()

                if compression_loss_type == "focal":
                    focal_loss_fn = FocalLoss(
                        alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
                    )
                    sample_loss = focal_loss_fn(sample_logits, sample_labels)
                else:  # bce
                    sample_loss = F.binary_cross_entropy_with_logits(
                        sample_logits, sample_labels, reduction="mean"
                    )
                sample_losses.append(sample_loss)

            if len(sample_losses) > 0:
                # 对batch内的样本取平均
                compress_loss = torch.stack(sample_losses).mean()
                pos_rate = total_pos_tokens / max(total_valid_tokens, 1)
            else:
                compress_loss = torch.tensor(0.0, device=device)
                pos_rate = 0.0
        else:
            # 【全局聚合】原始行为：对所有valid token直接取全局平均
            logits_valid = token_logits[valid_mask]
            labels_valid = token_labels[valid_mask].float()

            if compression_loss_type == "focal":
                focal_loss_fn = FocalLoss(
                    alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
                )
                compress_loss = focal_loss_fn(logits_valid, labels_valid)
            else:  # bce
                compress_loss = F.binary_cross_entropy_with_logits(
                    logits_valid, labels_valid, reduction="mean"
                )

            pos_rate = float(labels_valid.mean().detach().cpu().item())

    # Score loss: score_logits are log probs (yes), convert to probability
    score_probs = torch.exp(score_logits)

    score_loss = F.mse_loss(score_probs, ground_truth_scores)

    # 组合损失
    total_loss = compress_loss * (1.0 - lambda_score) + score_loss * lambda_score

    logs = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "compress_loss": float(compress_loss.detach().cpu().item()),
        "score_loss": float(score_loss.detach().cpu().item()),
        "pos_rate": pos_rate,
    }
    return total_loss, logs


def collate_fn(batch):
    """Custom collate to stack all tensors"""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    doc_mask = torch.stack([b["doc_mask"] for b in batch])
    token_labels = torch.stack([b["token_labels"] for b in batch])
    scores = torch.stack([b["score"] for b in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "doc_mask": doc_mask,
        "token_labels": token_labels,
        "scores": scores,
    }


def evaluate(
    model: TokenScorer,
    dataloader,
    threshold: float = 0.5,
    lambda_score: float = 0.05,
    device: torch.device = None,
    rank: int = 0,
    compression_loss_type: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate model and return metrics using torchmetrics for DDP compatibility

    Returns:
        Dictionary containing:
        - loss, compress_loss, score_loss: float values
        - accuracy, f1, precision, recall: float values
        - confusion_matrix: 2x2 numpy array [[TN, FP], [FN, TP]]
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Initialize torchmetrics on the correct device
    accuracy_metric = torchmetrics.Accuracy(task="binary", threshold=threshold).to(
        device
    )
    f1_metric = torchmetrics.F1Score(task="binary", threshold=threshold).to(device)
    precision_metric = torchmetrics.Precision(task="binary", threshold=threshold).to(
        device
    )
    recall_metric = torchmetrics.Recall(task="binary", threshold=threshold).to(device)
    confusion_matrix_metric = torchmetrics.ConfusionMatrix(
        task="binary", threshold=threshold
    ).to(device)

    total_loss = 0.0
    total_compress_loss = 0.0
    total_score_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Use smaller dtype and clear cache to avoid OOM
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss, logs = compute_combined_loss(
                    model,
                    batch,
                    lambda_score=lambda_score,
                    device=device,
                    compression_loss_type=compression_loss_type,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                )

            total_loss += logs["total_loss"]
            total_compress_loss += logs["compress_loss"]
            total_score_loss += logs["score_loss"]
            num_samples += 1

            # For compression metrics, extract token logits
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                token_logits = outputs["token_logits"].float()

            attention_mask = batch["attention_mask"].to(device)
            doc_mask = batch["doc_mask"].to(device).bool()
            token_labels = batch["token_labels"].to(device)

            # 只在有效位置计算
            valid_mask = doc_mask & attention_mask.bool() & (token_labels != -100)

            if valid_mask.sum() > 0:
                logits_valid = token_logits[valid_mask]
                labels_valid = token_labels[valid_mask].float()

                # Update metrics with probabilities and labels
                probs = torch.sigmoid(logits_valid)
                accuracy_metric.update(probs, labels_valid.long())
                f1_metric.update(probs, labels_valid.long())
                precision_metric.update(probs, labels_valid.long())
                recall_metric.update(probs, labels_valid.long())
                confusion_matrix_metric.update(probs, labels_valid.long())

            # Clear cache after each batch to free up memory
            torch.cuda.empty_cache()

    # Calculate metrics (synchronized across processes if DDP)
    avg_loss = total_loss / max(num_samples, 1)
    avg_compress_loss = total_compress_loss / max(num_samples, 1)
    avg_score_loss = total_score_loss / max(num_samples, 1)
    accuracy = accuracy_metric.compute().item()
    f1 = f1_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    confusion_matrix = confusion_matrix_metric.compute().cpu().numpy()

    # Reset metrics for next evaluation
    accuracy_metric.reset()
    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    confusion_matrix_metric.reset()

    # Final cache clear
    torch.cuda.empty_cache()

    return {
        "loss": avg_loss,
        "compress_loss": avg_compress_loss,
        "score_loss": avg_score_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion_matrix.tolist(),  # Convert to list for JSON serialization
    }


def save_eval_with_token_scores(
    model,
    dataset: CodePruneDataset,
    val_indices: List[int],
    tokenizer: AutoTokenizer,
    out_path: str,
    device: torch.device,
    rank: int = 0,
    max_dataset_size: int = 100,
):
    """Save eval set with per-code-token scores to a JSONL file.

    Each line contains the original data fields plus `token_scores`:
    list of [token_str, score] for every token in the code (in sequence order).
    Only executed by the main process (caller should guard by rank==0).
    """
    # Only main process writes
    if rank != 0:
        return

    model.eval()
    out_path = str(out_path)
    small_vis_dataset = val_indices[:max_dataset_size]
    with open(out_path, "w", encoding="utf-8") as fo:
        for idx in tqdm(small_vis_dataset, desc="Saving eval token scores"):
            # HINT: no batch for simplicity
            # original data
            item: DictData = dataset.data[idx]

            # build the same input as dataset would
            sample = dataset[idx]

            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            doc_mask = sample["doc_mask"]  # on CPU

            with torch.no_grad():
                # Use fp16 autocast to reduce memory usage during inference
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                token_logits = outputs["token_logits"].float()  # [1, L]
                score_logits = outputs["score_logits"].float()  # [1]

            # token_logits: [1, L] -> squeeze
            logits = token_logits.squeeze(0).cpu()
            probs = torch.sigmoid(logits)

            score_prob = score_logits.squeeze(0).cpu()
            predicted_score = float(torch.exp(score_prob).item())

            # extract code token positions from doc_mask
            code_positions = doc_mask.bool().nonzero(as_tuple=True)[0].tolist()

            code_token_ids = [
                int(sample["input_ids"][pos].item()) for pos in code_positions
            ]
            # convert ids to tokens (batch)
            tokens = tokenizer.convert_ids_to_tokens(code_token_ids)

            token_scores = []
            for tkn, pos in zip(tokens, code_positions):
                score = float(probs[pos].item())
                token_scores.append([tkn, score])

            # Clear GPU cache after each sample
            torch.cuda.empty_cache()

            out_obj = {
                "query": item.query,
                "code": item.code,
                "kept_frags": item.kept_frags,
                "score": item.score,
                "predicted_score": predicted_score,
                "token_scores": token_scores,
            }
            fo.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    console.print(f"Saved eval token scores to {out_path}")


def export_attention_data(
    model,
    dataset: CodePruneDataset,
    indices: List[int],
    tokenizer: AutoTokenizer,
    out_path: str,
    device: torch.device,
    rank: int = 0,
    max_dataset_size: int = 100,
):
    """Export attention weights and layer features for visualization.

    Each line contains:
    - query, code: original data
    - doc_start, doc_end: code token positions in the sequence
    - attention_weights: list of attention matrices for each fusion layer
    - early_attention, middle_attention, final_attention: average attention per token
      (only if use_multi_layer_fusion is True)
    """
    if rank != 0:
        return

    model.eval()
    actual_model = model.module if hasattr(model, "module") else model
    out_path = str(out_path)
    small_dataset = indices[:max_dataset_size]

    with open(out_path, "w", encoding="utf-8") as fo:
        for idx in tqdm(small_dataset, desc="Exporting attention data"):
            item: DictData = dataset.data[idx]
            sample = dataset[idx]

            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            doc_mask = sample["doc_mask"]

            # Get doc_start and doc_end
            doc_positions = doc_mask.bool().nonzero(as_tuple=True)[0].tolist()
            if not doc_positions:
                continue
            doc_start = doc_positions[0]
            doc_end = doc_positions[-1] + 1

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_attention=True,
                    )

            # Extract attention weights
            attention_weights_list = outputs.get("attention_weights", [])

            # Convert attention weights to lists (average over heads if multi-head)
            attention_data = []
            for attn_weights in attention_weights_list:
                if attn_weights.dim() == 4:  # [B, num_heads, L, L]
                    attn_weights = attn_weights.mean(dim=1)  # Average over heads
                # [B, L, L] -> [L, L] -> extract code part
                attn = attn_weights[0].cpu().numpy()  # [L, L]
                code_attn = attn[
                    doc_start:doc_end, doc_start:doc_end
                ]  # [doc_len, doc_len]
                # Average attention: for each code token, average over all positions it attends to
                avg_attn = code_attn.mean(axis=1).tolist()  # [doc_len]
                attention_data.append(avg_attn)

            # If multi-layer fusion, compute separate attention for each layer
            early_attention = None
            middle_attention = None
            final_attention = None

            if actual_model.use_multi_layer_fusion and len(attention_weights_list) > 0:
                # Use the first fusion layer's attention as approximation
                # In practice, we compute attention on the fused features
                # For visualization, we can use the fusion attention as proxy
                if attention_weights_list:
                    fusion_attn = attention_weights_list[0]
                    if fusion_attn.dim() == 4:
                        fusion_attn = fusion_attn.mean(dim=1)
                    fusion_attn = fusion_attn[0].cpu().numpy()
                    code_fusion_attn = fusion_attn[doc_start:doc_end, doc_start:doc_end]
                    avg_fusion = code_fusion_attn.mean(axis=1).tolist()
                    # Use same attention for all layers (since fusion combines them)
                    early_attention = avg_fusion
                    middle_attention = avg_fusion
                    final_attention = avg_fusion

            # Get token offsets for visualization
            code_token_ids = input_ids[0][doc_start:doc_end].cpu().tolist()
            enc = tokenizer(
                item.code, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets = enc["offset_mapping"][: len(code_token_ids)]

            out_obj = {
                "query": item.query,
                "code": item.code,
                "doc_start": int(doc_start),
                "doc_end": int(doc_end),
                "token_offsets": [[int(start), int(end)] for start, end in offsets],
                "attention_weights": attention_data,  # List of [doc_len] arrays
            }

            if early_attention is not None:
                out_obj["early_attention"] = early_attention
                out_obj["middle_attention"] = middle_attention
                out_obj["final_attention"] = final_attention

            fo.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()

    console.print(f"Saved attention data to {out_path}")


def train_epoch(
    model: TokenScorer,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    lambda_score: float = 0.05,
    rank: int = 0,
    compression_loss_type: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    use_sample_level_aggregation: bool = True,
) -> int:
    """Train for one epoch"""
    model.train()
    device = next(model.parameters()).device

    # Only show progress bar on main process
    if is_main_process(rank):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for batch in pbar:
        optimizer.zero_grad()

        batch_loss, logs = compute_combined_loss(
            model,
            batch,
            lambda_score=lambda_score,
            device=device,
            compression_loss_type=compression_loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            use_sample_level_aggregation=use_sample_level_aggregation,
        )

        batch_loss.backward()
        optimizer.step()
        scheduler.step()  # Step scheduler after each batch

        global_step += 1

        # Log to tensorboard (only on main process)
        if is_main_process(rank):
            writer.add_scalar("train/loss_step", logs["total_loss"], global_step)
            writer.add_scalar(
                "train/compress_loss_step", logs["compress_loss"], global_step
            )
            writer.add_scalar("train/score_loss_step", logs["score_loss"], global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            # Update progress bar
            if isinstance(pbar, tqdm):
                pbar.set_postfix(
                    {
                        "loss": f"{logs['total_loss']:.4f}",
                        "c_loss": f"{logs['compress_loss']:.4f}",
                        "s_loss": f"{logs['score_loss']:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

    return global_step


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> TokenScorer:
    """Load a TokenScorer model from checkpoint path.

    Args:
        checkpoint_path: Path to the model checkpoint directory or .pt file
        model_name: Base model name
        tokenizer: Tokenizer instance
        device: Device to load model on

    Returns:
        Loaded TokenScorer model
    """
    import os

    # Determine config and weights paths
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "model_config.json")
        weights_path = os.path.join(checkpoint_path, "best_model.pt")
    else:
        # Assume checkpoint_path is the weights file
        weights_path = checkpoint_path
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "model_config.json")

    # Load config
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        console.print(f"Loaded config from {config_path}")
    else:
        # Use default config if not found
        console.print(f"[yellow]Config file not found at {config_path}, using default config[/yellow]")
        config = {
            "bottleneck": 256,
            "dropout": 0.1,
            "num_finetune_layers": 2,
            "num_fusion_layers": 1,
            "num_heads": 8,
            "use_multi_layer_fusion": False,
            "early_layer_ratio": 0.25,
            "middle_layer_ratio": 0.5,
            "compression_head_type": "simple",
        }

    # Create model
    model = TokenScorer(
        model_name=model_name,
        tokenizer=tokenizer,
        bottleneck=config.get("bottleneck", 256),
        dropout=config.get("dropout", 0.1),
        num_finetune_layers=config.get("num_finetune_layers", 0),
        num_fusion_layers=config.get("num_fusion_layers", 1),
        num_heads=config.get("num_heads", 8),
        use_multi_layer_fusion=config.get("use_multi_layer_fusion", False),
        early_layer_ratio=config.get("early_layer_ratio", 0.25),
        middle_layer_ratio=config.get("middle_layer_ratio", 0.5),
        compression_head_type=config.get("compression_head_type", "ffn"),
    )

    # Load weights
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        console.print(f"Loaded weights from {weights_path}")
    else:
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    model = model.to(device)
    model.eval()
    return model


def evaluate_multiple_models(
    model_paths: List[str],
    eval_dataloader,
    model_name: str,
    tokenizer: AutoTokenizer,
    threshold: float,
    device: torch.device,
    rank: int,
) -> List[Dict[str, Any]]:
    """Evaluate multiple models and return their metrics.

    Each model's configuration (lambda_score, compression_loss_type, etc.)
    is loaded from its own model_config.json file.

    Args:
        model_paths: List of paths to model checkpoints
        eval_dataloader: DataLoader for evaluation
        model_name: Base model name
        tokenizer: Tokenizer instance
        threshold: Classification threshold
        device: Device to run evaluation on
        rank: Process rank

    Returns:
        List of dictionaries containing model path and metrics
    """
    results = []

    for model_path in model_paths:
        if is_main_process(rank):
            console.print(f"\n[bold]{'=' * 60}[/bold]")
            console.print(f"Evaluating model: {model_path}")
            console.print(f"[bold]{'=' * 60}[/bold]")

        try:
            # Load model
            model = load_model_from_checkpoint(
                checkpoint_path=model_path,
                model_name=model_name,
                tokenizer=tokenizer,
                device=device,
            )

            # Load model-specific config for evaluation parameters
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "model_config.json")
            else:
                config_dir = os.path.dirname(model_path)
                config_path = os.path.join(config_dir, "model_config.json")

            # Read config to get model-specific eval parameters
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                lambda_score = config.get("lambda_score", 0.05)
                compression_loss_type = config.get("compression_loss_type", "bce")
                focal_alpha = config.get("focal_alpha", 0.25)
                focal_gamma = config.get("focal_gamma", 2.0)

                if is_main_process(rank):
                    console.print(
                        f"Using model config: lambda_score={lambda_score}, "
                        f"compression_loss_type={compression_loss_type}, "
                        f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}"
                    )
            else:
                console.print(
                    f"[yellow]Config not found at {config_path}, using default eval parameters[/yellow]"
                )
                lambda_score = 0.05
                compression_loss_type = "bce"
                focal_alpha = 0.25
                focal_gamma = 2.0

            # Evaluate
            metrics = evaluate(
                model=model,
                dataloader=eval_dataloader,
                threshold=threshold,
                lambda_score=lambda_score,
                device=device,
                rank=rank,
                compression_loss_type=compression_loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )

            # Add model path to results
            result = {"model_path": model_path, **metrics}
            results.append(result)

            if is_main_process(rank):
                console.print(
                    f"Results - Loss: {metrics['loss']:.4f}, "
                    f"C_Loss: {metrics['compress_loss']:.4f}, "
                    f"S_Loss: {metrics['score_loss']:.4f}, "
                    f"Acc: {metrics['accuracy']:.4f}, "
                    f"Prec: {metrics['precision']:.4f}, "
                    f"Rec: {metrics['recall']:.4f}, "
                    f"F1: {metrics['f1']:.4f}"
                )

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            console.print(f"[red]Failed to evaluate model {model_path}: {str(e)}[/red]")
            import traceback

            traceback.print_exc()

    return results


def print_comparison_summary(results: List[Dict[str, Any]]):
    """Print a formatted comparison table of evaluation results.

    Args:
        results: List of dictionaries containing model paths and metrics
    """
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    console.print("\n" + "=" * 140)
    console.print("MODEL COMPARISON SUMMARY")
    console.print("=" * 140)

    headers = [
        "Model Path",
        "Loss",
        "C_Loss",
        "S_Loss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]
    col_widths = [40, 10, 10, 10, 10, 10, 10, 10]

    header_row = ""
    for header, width in zip(headers, col_widths):
        header_row += f"{header:<{width}}"
    console.print(header_row)
    console.print("-" * 140)

    for result in results:
        mname = os.path.basename(result["model_path"])
        if len(mname) > 38:
            mname = "..." + mname[-35:]

        row = f"{mname:<40}"
        row += f"{result['loss']:<10.4f}"
        row += f"{result['compress_loss']:<10.4f}"
        row += f"{result['score_loss']:<10.4f}"
        row += f"{result['accuracy']:<10.4f}"
        row += f"{result['precision']:<10.4f}"
        row += f"{result['recall']:<10.4f}"
        row += f"{result['f1']:<10.4f}"
        console.print(row)

    console.print("=" * 140)

    # Find and highlight best models
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
    best_acc_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])
    best_precision_idx = max(range(len(results)), key=lambda i: results[i]["precision"])
    best_recall_idx = max(range(len(results)), key=lambda i: results[i]["recall"])

    console.print(
        f"\nBest F1: {results[best_f1_idx]['f1']:.4f} - {os.path.basename(results[best_f1_idx]['model_path'])}"
    )
    console.print(
        f"Best Accuracy: {results[best_acc_idx]['accuracy']:.4f} - {os.path.basename(results[best_acc_idx]['model_path'])}"
    )
    console.print(
        f"Best Precision: {results[best_precision_idx]['precision']:.4f} - {os.path.basename(results[best_precision_idx]['model_path'])}"
    )
    console.print(
        f"Best Recall: {results[best_recall_idx]['recall']:.4f} - {os.path.basename(results[best_recall_idx]['model_path'])}"
    )

    console.print("\n" + "=" * 140)
    console.print("CONFUSION MATRICES")
    console.print("=" * 140)
    for result in results:
        mname = os.path.basename(result["model_path"])
        cm = result["confusion_matrix"]
        console.print(f"\n{mname}:")
        console.print(f"  [[TN={cm[0][0]:.0f}, FP={cm[0][1]:.0f}],")
        console.print(f"   [FN={cm[1][0]:.0f}, TP={cm[1][1]:.0f}]]")

        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        if total > 0:
            console.print(f"  Total predictions: {total:.0f}")
            console.print(f"  Positive rate: {(tp + fp) / total * 100:.2f}%")
            console.print(f"  Negative rate: {(tn + fn) / total * 100:.2f}%")

    console.print("=" * 140 + "\n")


train_app = typer.Typer(help="Train LLM token scorer for code pruning")


@train_app.command()
def main(
    input_file: str = typer.Option(..., "-i", "--input-file", help="Path to input data file (JSONL)"),
    model_name: str = typer.Option(..., "--model-name", help="Base model name or path (e.g. Qwen/Qwen3-Reranker-0.6B)"),
    hidden_size: int = typer.Option(256, "--hidden-size"),
    dropout: float = typer.Option(0.1, "--dropout"),
    batch_size: int = typer.Option(4, "--batch-size"),
    epochs: int = typer.Option(2, "--epochs"),
    lr: float = typer.Option(1e-4, "--lr"),
    warmup_ratio: float = typer.Option(0.1, "--warmup-ratio"),
    threshold: float = typer.Option(0.5, "--threshold"),
    train_split: float = typer.Option(0.9, "--train-split"),
    log_dir: str = typer.Option("llm_experiments/token_scorer", "--log-dir"),
    seed: int = typer.Option(42, "--seed"),
    instruction: str = typer.Option(
        "Given a query, judge if the document(code) is related to query.",
        "--instruction",
    ),
    num_finetune_layers: int = typer.Option(0, "--num-finetune-layers"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    num_fusion_layers: int = typer.Option(1, "--num-fusion-layers"),
    num_heads: int = typer.Option(8, "--num-heads"),
    lambda_score: float = typer.Option(0.05, "--lambda-score"),
    compression_head_type: str = typer.Option("ffn", "--compression-head-type"),
    compression_loss_type: str = typer.Option("focal", "--compression-loss-type"),
    focal_alpha: float = typer.Option(0.25, "--focal-alpha"),
    auto_focal_alpha: bool = typer.Option(False, "--auto-focal-alpha"),
    focal_gamma: float = typer.Option(2.0, "--focal-gamma"),
    use_sample_level_aggregation: bool = typer.Option(True, "--use-sample-level-aggregation"),
    no_sample_level_aggregation: bool = typer.Option(False, "--no-sample-level-aggregation"),
    use_multi_layer_fusion: bool = typer.Option(False, "--use-multi-layer-fusion"),
    early_layer_ratio: float = typer.Option(0.25, "--early-layer-ratio"),
    middle_layer_ratio: float = typer.Option(0.5, "--middle-layer-ratio"),
    eval_only: bool = typer.Option(False, "--eval-only"),
    eval_dataset: Optional[str] = typer.Option(None, "--eval-dataset"),
    model_paths: Optional[List[str]] = typer.Option(None, "--model-paths"),
    export_attention: Optional[str] = typer.Option(None, "--export-attention"),
    attention_dataset: Optional[str] = typer.Option(None, "--attention-dataset"),
    max_attention_samples: int = typer.Option(100, "--max-attention-samples"),
):
    args = type("Args", (), {
        "input_file": input_file,
        "model_name": model_name,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "threshold": threshold,
        "train_split": train_split,
        "log_dir": log_dir,
        "seed": seed,
        "instruction": instruction,
        "num_finetune_layers": num_finetune_layers,
        "weight_decay": weight_decay,
        "num_fusion_layers": num_fusion_layers,
        "num_heads": num_heads,
        "lambda_score": lambda_score,
        "compression_head_type": compression_head_type,
        "compression_loss_type": compression_loss_type,
        "focal_alpha": focal_alpha,
        "auto_focal_alpha": auto_focal_alpha,
        "focal_gamma": focal_gamma,
        "use_sample_level_aggregation": use_sample_level_aggregation,
        "no_sample_level_aggregation": no_sample_level_aggregation,
        "use_multi_layer_fusion": use_multi_layer_fusion,
        "early_layer_ratio": early_layer_ratio,
        "middle_layer_ratio": middle_layer_ratio,
        "eval_only": eval_only,
        "eval_dataset": eval_dataset,
        "model_paths": model_paths,
        "export_attention": export_attention,
        "attention_dataset": attention_dataset,
        "max_attention_samples": max_attention_samples,
    })()

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()

    # Set device
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.padding_side = "left"
    if is_main_process(rank):
        console.print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

        if args.eval_only:
            if args.model_paths is None or len(args.model_paths) == 0:
                console.print("[red]--eval-only requires --model-paths to be specified[/red]")
                cleanup_ddp()
                exit(1)
            if args.eval_dataset is None:
                console.print("[red]--eval-only requires --eval-dataset to be specified[/red]")
                cleanup_ddp()
                exit(1)
            console.print("Running in EVALUATION-ONLY mode")
            console.print(f"Eval dataset: {args.eval_dataset}")
            console.print(f"Models to evaluate: {len(args.model_paths)}")
        else:
            console.print(f"Loading training data from {args.input_file}")

    # Load data based on mode
    if args.eval_only:
        # Evaluation-only mode: load eval dataset
        eval_data: List[DictData] = []
        with open(args.eval_dataset, "r") as f:
            for i, line in enumerate(f):
                try:
                    eval_data.append(DictData(**json.loads(line)))
                except json.JSONDecodeError:
                    if is_main_process(rank):
                        console.print(f"[yellow]Skipping line {i}: JSON decode error[/yellow]")
                    continue
                except Exception:
                    continue

        if is_main_process(rank):
            console.print(f"Loaded {len(eval_data)} evaluation samples")

        data = eval_data  # Use eval data as main data
    else:
        # Training mode: load training data
        data: List[DictData] = []
        with open(args.input_file, "r") as f:
            for i, line in enumerate(f):
                try:
                    data.append(DictData(**json.loads(line)))
                except json.JSONDecodeError:
                    if is_main_process(rank):
                        console.print(f"[yellow]Skipping line {i}: JSON decode error[/yellow]")
                    continue
                except Exception:
                    continue

        if is_main_process(rank):
            console.print(f"Loaded {len(data)} samples")

    # 确定是否计算类别统计（仅在非eval-only模式下且需要自动alpha时计算）
    compute_class_ratio = not args.eval_only and args.auto_focal_alpha

    dataset = CodePruneDataset(
        data,
        tokenizer,
        max_length=8192,
        instruction=args.instruction,
        compute_class_ratio=compute_class_ratio,
    )

    # 处理自动focal alpha
    use_sample_level_aggregation = (
        args.use_sample_level_aggregation and not args.no_sample_level_aggregation
    )

    if args.auto_focal_alpha and dataset.auto_focal_alpha is not None:
        effective_focal_alpha = dataset.auto_focal_alpha
        if is_main_process(rank):
            console.print(f"Using auto-computed focal_alpha: {effective_focal_alpha:.4f}")
    else:
        effective_focal_alpha = args.focal_alpha
        if is_main_process(rank) and not args.eval_only:
            console.print(f"Using manual focal_alpha: {effective_focal_alpha:.4f}")

    if is_main_process(rank):
        console.print(f"Sample-level loss aggregation: {use_sample_level_aggregation}")

    # Handle attention export mode (before creating dataset)
    if args.export_attention:
        if args.model_paths is None or len(args.model_paths) == 0:
            console.print("[red]--export-attention requires --model-paths to be specified[/red]")
            cleanup_ddp()
            exit(1)

        # Load dataset for attention export
        attention_dataset_path = (
            args.attention_dataset or args.eval_dataset or args.input_file
        )
        if not attention_dataset_path:
            console.print(
                "[red]--export-attention requires --attention-dataset or --eval-dataset or --input-file[/red]"
            )
            cleanup_ddp()
            exit(1)

        attention_data: List[DictData] = []
        with open(attention_dataset_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    attention_data.append(DictData(**json.loads(line)))
                except json.JSONDecodeError:
                    if is_main_process(rank):
                        console.print(f"[yellow]Skipping line {i}: JSON decode error[/yellow]")
                    continue
                except Exception:
                    continue

        if is_main_process(rank):
            console.print(f"Loaded {len(attention_data)} samples for attention export")

        # Create dataset
        attention_dataset = CodePruneDataset(
            attention_data,
            tokenizer,
            max_length=8192,
            instruction=args.instruction,
            compute_class_ratio=False,
        )

        # Export attention for each model
        for model_path in args.model_paths:
            if is_main_process(rank):
                console.print(f"\n[bold]{'=' * 60}[/bold]")
                console.print(f"Exporting attention for model: {model_path}")
                console.print(f"[bold]{'=' * 60}[/bold]")

            # Load model
            model = load_model_from_checkpoint(
                checkpoint_path=model_path,
                model_name=model_name,
                tokenizer=tokenizer,
                device=device,
            )

            if world_size > 1:
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)

            # Export attention data
            output_path = args.export_attention
            if len(args.model_paths) > 1:
                # Add model name to output path if multiple models
                model_name_suffix = (
                    os.path.basename(model_path).replace(".pt", "").replace("/", "_")
                )
                base_path = os.path.splitext(output_path)[0]
                ext = os.path.splitext(output_path)[1]
                output_path = f"{base_path}_{model_name_suffix}{ext}"

            export_attention_data(
                model=model,
                dataset=attention_dataset,
                indices=list(range(len(attention_dataset))),
                tokenizer=tokenizer,
                out_path=output_path,
                device=device,
                rank=rank,
                max_dataset_size=args.max_attention_samples,
            )

            del model
            torch.cuda.empty_cache()

        cleanup_ddp()
        exit(0)

    # Handle eval-only mode
    if args.eval_only:
        # In eval-only mode, use the entire dataset for evaluation
        if is_main_process(rank):
            console.print(f"Eval dataset size: {len(dataset)}")

        # Create sampler for DDP
        if world_size > 1:
            eval_sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            eval_sampler = None

        # Create eval dataloader
        eval_batch_size = max(1, args.batch_size // 4)
        eval_loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            sampler=eval_sampler,
            collate_fn=collate_fn,
            pin_memory=False,
            num_workers=2,
        )

        # Evaluate all models
        if is_main_process(rank):
            console.print("\n" + "=" * 60)
            console.print("Starting multi-model evaluation")
            console.print("=" * 60 + "\n")

        results = evaluate_multiple_models(
            model_paths=args.model_paths,
            eval_dataloader=eval_loader,
            model_name=model_name,
            tokenizer=tokenizer,
            threshold=args.threshold,
            device=device,
            rank=rank,
        )

        # Print comparison summary
        if is_main_process(rank):
            print_comparison_summary(results)

            # Save results to JSON file
            results_path = f"{args.log_dir}/multi_model_comparison.json"
            os.makedirs(args.log_dir, exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            console.print(f"Saved comparison results to {results_path}")

        # Cleanup and exit
        cleanup_ddp()
        exit(0)

    # Training mode: Split train/val
    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if is_main_process(rank):
        console.print(f"Train size: {train_size}, Val size: {val_size}")

    # Create samplers for DDP
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=2,
    )

    # Use smaller batch size for evaluation to avoid OOM (especially important for long sequences)
    eval_batch_size = max(1, args.batch_size // 4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=2,
    )

    if is_main_process(rank):
        console.print(f"Initializing model: {model_name}")
        console.print(
            f"Finetuning last {args.num_finetune_layers} layers, Weight decay: {args.weight_decay}"
        )

    scorer = TokenScorer(
        model_name=model_name,
        tokenizer=tokenizer,
        bottleneck=args.hidden_size,
        dropout=args.dropout,
        num_finetune_layers=args.num_finetune_layers,
        num_fusion_layers=args.num_fusion_layers,
        num_heads=args.num_heads,
        use_multi_layer_fusion=args.use_multi_layer_fusion,
        early_layer_ratio=args.early_layer_ratio,
        middle_layer_ratio=args.middle_layer_ratio,
        compression_head_type=args.compression_head_type,  # 新增
    )
    scorer = scorer.to(device)

    # Wrap model with DDP
    if world_size > 1:
        scorer = DDP(
            scorer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        model_without_ddp = scorer.module
    else:
        model_without_ddp = scorer

    # Count trainable parameters
    total_params = sum(p.numel() for p in scorer.parameters())
    trainable_params = sum(p.numel() for p in scorer.parameters() if p.requires_grad)

    if is_main_process(rank):
        console.print(
            f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}"
        )

    # Separate parameters into backbone and head
    backbone_params = list(model_without_ddp.backbone.parameters())
    backbone_param_ids = set(id(p) for p in backbone_params)
    head_params = [
        p for p in model_without_ddp.parameters() if id(p) not in backbone_param_ids
    ]

    # Filter for trainable parameters only
    backbone_params_trainable = [p for p in backbone_params if p.requires_grad]
    head_params_trainable = [p for p in head_params if p.requires_grad]

    # Create parameter groups with different weight decay
    param_groups = [
        {
            "params": backbone_params_trainable,
            "lr": args.lr,
            "weight_decay": args.weight_decay,  # Apply weight decay to backbone
        },
        {
            "params": head_params_trainable,
            "lr": args.lr,
            "weight_decay": 0.0,  # HINT: No weight decay for classification head
        },
    ]

    optimizer = torch.optim.AdamW(param_groups)  # HINT: change to Muon optimizer

    # Calculate total training steps
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    # Create cosine scheduler with linear warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if is_main_process(rank):
        console.print(f"Total training steps: {num_training_steps}")
        console.print(
            f"Warmup steps: {num_warmup_steps} ({args.warmup_ratio * 100:.1f}%)"
        )

    writer = SummaryWriter(args.log_dir) if is_main_process(rank) else None
    if is_main_process(rank):
        console.print(f"Logging to {args.log_dir}")

    global_step = 0
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        if is_main_process(rank):
            console.print(f"=== Epoch {epoch}/{args.epochs} ===")

        # Set epoch for sampler (important for proper shuffling in DDP)
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # Train
        global_step = train_epoch(
            scorer,
            train_loader,
            optimizer,
            scheduler,
            epoch,
            writer,
            global_step,
            args.lambda_score,
            rank,
            args.compression_loss_type,
            effective_focal_alpha,
            args.focal_gamma,
            use_sample_level_aggregation,
        )

        # Evaluate
        if is_main_process(rank):
            console.print("Evaluating on validation set...")

        val_metrics = evaluate(
            scorer,
            val_loader,
            threshold=args.threshold,
            lambda_score=args.lambda_score,
            device=device,
            rank=rank,
            compression_loss_type=args.compression_loss_type,
            focal_alpha=effective_focal_alpha,
            focal_gamma=args.focal_gamma,
        )

        # Log to tensorboard (only on main process)
        if is_main_process(rank):
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/compress_loss", val_metrics["compress_loss"], epoch)
            writer.add_scalar("val/score_loss", val_metrics["score_loss"], epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/precision", val_metrics["precision"], epoch)
            writer.add_scalar("val/recall", val_metrics["recall"], epoch)
            writer.add_scalar("val/f1", val_metrics["f1"], epoch)

            # Log to terminal
            console.print(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"C_Loss: {val_metrics['compress_loss']:.4f}, "
                f"S_Loss: {val_metrics['score_loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"Prec: {val_metrics['precision']:.4f}, "
                f"Rec: {val_metrics['recall']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}"
            )

            # Save best model
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                # Also save eval set with per-code-token scores for analysis
                eval_out = f"{args.log_dir}/eval_with_token_scores.jsonl"
                save_eval_with_token_scores(
                    model_without_ddp,
                    dataset,
                    val_dataset.indices,
                    tokenizer,
                    eval_out,
                    device,
                    rank,
                )

                torch.save(
                    model_without_ddp.state_dict(), f"{args.log_dir}/best_model.pt"
                )

                # Save model configuration to JSON file
                config_path = f"{args.log_dir}/model_config.json"
                model_config = {
                    "model_name": model_name,
                    "bottleneck": args.hidden_size,
                    "dropout": args.dropout,
                    "num_finetune_layers": args.num_finetune_layers,
                    "num_fusion_layers": args.num_fusion_layers,
                    "num_heads": args.num_heads,
                    "use_multi_layer_fusion": args.use_multi_layer_fusion,
                    "early_layer_ratio": args.early_layer_ratio,
                    "middle_layer_ratio": args.middle_layer_ratio,
                    "compression_head_type": args.compression_head_type,
                    "compression_loss_type": args.compression_loss_type,
                    "focal_alpha": effective_focal_alpha,
                    "focal_alpha_auto": args.auto_focal_alpha,
                    "focal_gamma": args.focal_gamma,
                    "lambda_score": args.lambda_score,
                    "use_sample_level_aggregation": use_sample_level_aggregation,
                }
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(model_config, f, indent=2, ensure_ascii=False)
                console.print(f"Saved best model with F1: {best_f1:.4f}")
                console.print(f"Saved model config to {config_path}")

    if is_main_process(rank):
        writer.close()
        console.print("Training complete!")

    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    train_app()
