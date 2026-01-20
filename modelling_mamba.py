import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

# Import your config (assuming it's in the same folder)
from configuration_mamba import MambaQuinConfig

# 1. THE BENTO BOX (Custom Output) 🍱
@dataclass
class MambaOutput(ModelOutput):
    loss: torch.FloatTensor | None      # 🚨 The Trainer needs this!
    logits: torch.FloatTensor = None              # Sentiment Scores
    reconstruction: torch.FloatTensor = None      # Autoencoder output
    latent_z: torch.FloatTensor = None            # The compressed embedding
    cluster_logits: torch.FloatTensor | None # For DEKM/Clustering

# 2. THE MODEL CLASS 🐍
class MambaQuin(PreTrainedModel):
    config_class = MambaQuinConfig # 🔗 Link to the blueprint

    def __init__(self, config):
        super().__init__(config)
        # (This downloads the pre-trained Mamba weights)
        self.backbone = AutoModel.from_pretrained(config.backbone_model)
        
        # Input size is 768 + 768 = 1536
        self.encoder = nn.Sequential(
            nn.Linear(config.d_model * 2, 256), 
            nn.ReLU(),
            nn.Linear(256, config.latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.d_model * 2) 
        )
        
        self.sentiment_head = nn.Linear(config.latent_dim, config.num_classes)
        
        # Randomly init 3 centers in the latent space
        self.cluster_centers = nn.Parameter(torch.randn(config.num_clusters, config.latent_dim))

        self.loss_fct_class = nn.CrossEntropyLoss()
        self.loss_fct_recon = nn.MSELoss()

        self.cls_loss_weights = config.cls_loss_weights
        self.rec_loss_weights = config.rec_loss_weights
        self.clt_loss_weights = config.clt_loss_weights

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 1. Run Mamba 🐍
        # Mamba usually ignores attention_mask, but we pass it just in case
        outputs = self.backbone(input_ids=input_ids)
        hidden = outputs.last_hidden_state # [Batch, Seq, 768]
        
        # 2. Pooling (The Sandwich Strategy) 🥪
        # Average Pooling (Vibe)
        mean_pool = torch.mean(hidden, dim=1)
        # Max Pooling (Drama)
        max_pool, _ = torch.max(hidden, dim=1)
        
        # CONCATENATE THEM -> [Batch, 1536]
        combo_tensor = torch.cat((mean_pool, max_pool), dim=1)
        
        # 3. Encode 📉
        z = self.encoder(combo_tensor)
        
        # 4. Heads 🐲
        logits = self.sentiment_head(z)
        reconstruction = self.decoder(z)
        
        # Calculates distance from every Z to every Center
        dist = torch.cdist(z, self.cluster_centers, p=2)
        q_logits = -dist # Closer = Higher Score

        loss_clt = q_logits
        
        total_loss = None
        if labels is not None:
            # A. Sentiment Loss
            loss_cls = self.loss_fct_class(logits, labels)
            
            # B. Reconstruction Loss (Try to rebuild the combo vector)
            loss_rec = self.loss_fct_recon(reconstruction, combo_tensor)
            
            # C. Combine them (Weighted)
            total_loss = (self.cls_loss_weights * loss_cls) + (self.rec_loss_weights * loss_rec) + (self.rec_loss_weights * loss_rec)

        # 6. Return the Bento Box 🍱
        return MambaOutput(
            loss=total_loss,
            logits=logits,
            reconstruction=reconstruction,
            latent_z=z,
            cluster_logits=q_logits
        )