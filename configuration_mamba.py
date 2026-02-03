from transformers import PretrainedConfig

class MambaQuinConfig(PretrainedConfig):
    model_type = "mamba_quin" 

    def __init__(
        self,
        d_model=768,        # Mamba hidden size (default for 130m)
        num_classes=2,      # Sentiment: Positive/Negative
        latent_dim=128,     # The size of the compressed brain (Z)
        backbone_model="state-spaces/mamba-130m-hf",
        encoder_layers = 2,
        decoder_layers = 2,
        sen_loss_weights = 0.5,
        rec_loss_weights = 1,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.backbone_model = backbone_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.sen_loss_weights = sen_loss_weights
        self.rec_loss_weights = rec_loss_weights
        super().__init__(**kwargs)
