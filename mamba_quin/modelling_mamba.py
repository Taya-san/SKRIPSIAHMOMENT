import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

# Import your config (assuming it's in the same folder)
from .configuration_mamba import MambaQuinConfig

# 1. THE BENTO BOX (Custom Output) 🍱
@dataclass
class MambaOutput(ModelOutput):
    loss: torch.FloatTensor | None = None      # 🚨 The Trainer needs this!
    logits: torch.FloatTensor = None              # Sentiment Scores
    reconstruction: torch.FloatTensor = None      # Autoencoder output
    latent_z: torch.FloatTensor = None            # The compressed embedding

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), 
                             nn.LayerNorm(self.latent_dim),
                             nn.GELU()
                             )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.output_dim), 
                             nn.LayerNorm(self.output_dim),
                             nn.GELU()
                             )
    
    def forward(self, x):
        return self.net(x)
        
class SentimentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), 
                             nn.LayerNorm(self.hidden_dim),
                             nn.GELU(),
                             nn.Dropout(0.1),
                             nn.Linear(self.hidden_dim, self.output_dim)
                             )
    
    def forward(self, x):
        return self.net(x)

# 2. THE MODEL CLASS 🐍
class MambaQuin(PreTrainedModel):
    config_class = MambaQuinConfig # 🔗 Link to the blueprint

    def __init__(self, config):
        super().__init__(config)
        # (This downloads the pre-trained Mamba weights)
        self.backbone = AutoModel.from_pretrained(config.backbone_model)
        
        encoder_layers = [Encoder(1536, config.latent_dim*config.encoder_layers)]

        for i in range(config.encoder_layers - 1):
            encoder_layers.append(Encoder(config.latent_dim*(i + 2), config.latent_dim*(i + 1)))

        decoder_layers = []

        for i in range(config.encoder_layers - 1):
            decoder_layers.append(Decoder(config.latent_dim*(i + 1), config.latent_dim*(i + 2)))
        
        decoder_layers.append(Decoder(config.latent_dim*config.decoder_layers, 1536))

        # Input size is 768 + 768 = 1536
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.sentiment_head = SentimentMLP(config.latent_dim, config.latent_dim, config.num_classes)


        self.loss_fct_class = nn.CrossEntropyLoss()
        self.loss_fct_recon = nn.MSELoss()

        self.cls_loss_weights = config.sen_loss_weights
        self.rec_loss_weights = config.rec_loss_weights

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 1. Run Mamba 🐍
        outputs = self.backbone(input_ids = input_ids,
                                attention_mask = attention_mask)
        hidden = outputs.last_hidden_state # [Batch, Seq, 768]
        att_mask = attention_mask.unsqueeze(-1).float()
        
        # Average Pooling (Vibe)
        sum_hidden_masked = torch.sum(hidden * att_mask, dim = 1)
        count_hidden_masked = att_mask.sum(dim=1, keepdim = True).clamp(min=1)
        mean_pool = sum_hidden_masked / count_hidden_masked

        # Max Pooling (Drama)
        hidden = hidden.masked_fill(att_mask == 0, -1e9)
        max_pool, _ = torch.max(hidden, dim=1)
        
        # CONCATENATE THEM -> [Batch, 1536]
        combo_tensor = torch.cat((mean_pool, max_pool), dim=1)
        
        # 3. Encode 📉
        z = self.encoder(combo_tensor)
        
        # 4. Heads 🐲
        logits = self.sentiment_head(z)
        reconstruction = self.decoder(z)
        
        total_loss = None
        if labels is not None:
            # A. Sentiment Loss
            loss_cls = self.loss_fct_class(logits, labels)
            
            # B. Reconstruction Loss (Try to rebuild the combo vector)
            loss_rec = self.loss_fct_recon(reconstruction, combo_tensor)
           
            # C. Combine them (Weighted)
            total_loss = (self.cls_loss_weights * loss_cls) + (self.rec_loss_weights * loss_rec)

        # 6. Return the Bento Box 🍱
        return MambaOutput(
            loss=total_loss,
            logits=logits,
            reconstruction=reconstruction,
            latent_z=z,
        )

