from transformers import PretrainedConfig

class MambaQuinConfig(PretrainedConfig):
    model_type = "mamba_quin" 

    def __init__(
        self,
        backbone_config,
        d_model=768,        # Mamba hidden size (default for 130m)
        num_classes=2,      # Sentiment: Positive/Negative
        latent_dim=128,     # The size of the compressed brain (Z)
        encoder_layers = 2,
        decoder_layers = 2,
        sen_loss_weights = 0.5,
        rec_loss_weights = 1,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.backbone_config = backbone_config.to_dict()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.sen_loss_weights = sen_loss_weights
        self.rec_loss_weights = rec_loss_weights
        super().__init__(**kwargs)
