from transformers import PretrainedConfig

class MambaQuinConfig(PretrainedConfig):
    model_type = "mamba_quin" 

    def __init__(
        self,
        d_model=768,        # Mamba hidden size (default for 130m)
        num_classes=2,      # Sentiment: Positive/Negative
        num_clusters=3,     # Clustering: How many groups?
        latent_dim=128,     # The size of the compressed brain (Z)
        backbone_model="state-spaces/mamba-130m-hf",
        cls_loss_weights = 0.45,
        rec_loss_weights = 0.45,
        clt_loss_weights = 0.1,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.backbone_model = backbone_model
        self.cls_loss_weights = cls_loss_weights
        self.rec_loss_weights = rec_loss_weights
        self.clt_loss_weights = clt_loss_weights
        super().__init__(**kwargs)