import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional

from .configuration import SwePrunerConfig
from .model_structure import TokenScorer


@dataclass
class SwePrunerOutput(ModelOutput):
    """
    Output class for SwePruner model.

    Args:
        token_logits: Token-level compression logits [batch_size, seq_len]
        score_logits: Document-level relevance score logits [batch_size]
    """

    # HINT: huggingface's ModelOutput has most one required field, so both are optional here, but in practice both are always returned
    token_logits: Optional[torch.FloatTensor] = None
    score_logits: Optional[torch.FloatTensor] = None


class SwePrunerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    config_class = SwePrunerConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights - DISABLED for debugging"""
        pass


class SwePrunerForCodeCompression(SwePrunerPreTrainedModel):
    """
    SwePruner model for code compression and relevance scoring.

    This wraps the TokenScorer from model_structure.py and provides HuggingFace-compatible interface.
    """

    def __init__(self, config: SwePrunerConfig, from_pretrained: bool = False):
        super().__init__(config)
        self.config = config

        # Check if we're loading from pretrained (HuggingFace sets config._name_or_path)
        # This is more reliable than relying on the from_pretrained parameter
        # because HuggingFace's from_pretrained doesn't pass this parameter directly
        is_loading_from_pretrained = from_pretrained or (
            hasattr(config, "_name_or_path") and config._name_or_path is not None
        )

        # Load tokenizer to pass to TokenScorer
        trust_remote_code = getattr(config, "trust_remote_code", True)

        tokenizer = AutoTokenizer.from_pretrained(
            config.backbone_model_name_or_path, trust_remote_code=trust_remote_code
        )
        load_pretrained_backbone = not is_loading_from_pretrained
        backbone_config = None

        if not load_pretrained_backbone:
            backbone_config = AutoConfig.from_pretrained(
                config.backbone_model_name_or_path, trust_remote_code=trust_remote_code
            )

        torch_dtype = getattr(config, "torch_dtype", None) or getattr(
            config, "dtype", None
        )
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, None)

        # Initialize TokenScorer with config parameters
        # The weights will be loaded from checkpoint if is_loading_from_pretrained is True
        self.model = TokenScorer(
            model_name=config.backbone_model_name_or_path,
            tokenizer=tokenizer,
            bottleneck=config.bottleneck,
            dropout=config.dropout,
            num_fusion_layers=config.num_fusion_layers,
            num_heads=config.num_heads,
            use_multi_layer_fusion=config.use_multi_layer_fusion,
            early_layer_ratio=config.early_layer_ratio,
            middle_layer_ratio=config.middle_layer_ratio,
            compression_head_type=config.compression_head_type,
            load_pretrained_backbone=load_pretrained_backbone,
            backbone_config=backbone_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

        # Store tokenizer reference
        self.tokenizer = tokenizer

        # Initialize weights (for newly added layers only, backbone is already initialized)
        # Skip post_init if loading from pretrained - we'll load the real weights
        if not is_loading_from_pretrained:
            self.post_init()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SwePrunerOutput:
        """
        Forward pass

        Args:
            input_ids: [B, L] input token ids
            attention_mask: [B, L] attention mask
            return_attention: Whether to return attention weights

        Returns:
            SwePrunerOutput with token_logits and score_logits
        """
        # Call TokenScorer forward
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Convert dict output to SwePrunerOutput
        return SwePrunerOutput(
            token_logits=outputs["token_logits"],
            score_logits=outputs["score_logits"],
        )

    def state_dict(self, *args, **kwargs):
        """Override state_dict to drop duplicated embedding references to the backbone."""
        state_dict = super().state_dict(*args, **kwargs)
        # Remove legacy word_embeddings pointer to avoid saving duplicated tensors
        if "model.word_embeddings" in state_dict:
            del state_dict["model.word_embeddings"]
        # embedding_layer shares weights with the backbone input embeddings
        if "model.embedding_layer.weight" in state_dict:
            del state_dict["model.embedding_layer.weight"]
        return state_dict

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save model to directory

        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments passed to parent save_pretrained
        """
        # Save config and model state dict
        # Use safe_serialization=False to avoid shared weight issues
        kwargs.setdefault("safe_serialization", False)
        super().save_pretrained(save_directory, **kwargs)
