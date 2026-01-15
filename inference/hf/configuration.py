from transformers import PretrainedConfig


class SwePrunerConfig(PretrainedConfig):
    """
    Configuration class for SwePruner model.

    SwePruner is a dual-head model for code compression and relevance scoring:
    - Compression head: Token-level scoring for code pruning
    - Scoring head: Document-level relevance score
    """

    model_type = "swepruner"

    def __init__(
        self,
        backbone_model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
        bottleneck: int = 256,
        dropout: float = 0.4,
        num_fusion_layers: int = 1,
        num_heads: int = 8,
        use_multi_layer_fusion: bool = True,
        early_layer_ratio: float = 0.25,
        middle_layer_ratio: float = 0.5,
        compression_head_type: str = "crf",
        compression_loss_type: str = "focal",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone_model_name_or_path = backbone_model_name_or_path
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.num_fusion_layers = num_fusion_layers
        self.num_heads = num_heads
        self.use_multi_layer_fusion = use_multi_layer_fusion
        self.early_layer_ratio = early_layer_ratio
        self.middle_layer_ratio = middle_layer_ratio
        self.compression_head_type = compression_head_type
        self.compression_loss_type = compression_loss_type
