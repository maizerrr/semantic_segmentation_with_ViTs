import torch
from torch import nn

from .encoder import build_encoder
from .decoder import build_decoder
from .aspp import build_aspp

class SwinDeepLab(nn.Module):
    def __init__(self, encoder_config, aspp_config, decoder_config):
        super().__init__()
        self.backbone = build_encoder(encoder_config)
        aspp = build_aspp(input_size=self.backbone.high_level_size,
                               input_dim=self.backbone.high_level_dim,
                               out_dim=self.backbone.low_level_dim, config=aspp_config)
        decoder = build_decoder(input_size=self.backbone.high_level_size,
                                     input_dim=self.backbone.low_level_dim,
                                     config=decoder_config)
        self.classifier = Classifier(aspp, decoder)

    def run_upsample(self, x):
        return self.upsample(x)

    def forward(self, x):
        low_level, high_level = self.backbone(x)
        x = self.classifier(low_level, high_level)
        return x

class Backbone(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        low_level, high_level = self.encoder(x)
        return low_level, high_level
    
class Classifier(nn.Module):
    def __init__(self, aspp, decoder):
        super().__init__()  
        self.aspp = aspp
        self.decoder = decoder
    
    def forward(self, low_level, high_level):
        x = self.aspp(high_level)
        x = self.decoder(low_level, x)
        return x