from .configuration_mamba import MambaQuinConfig
from .modelling_mamba import MambaQuin

from transformers import AutoConfig, AutoModel

AutoConfig.register("mamba_quin", MambaQuinConfig)
AutoModel.register(MambaQuinConfig, MambaQuin)
