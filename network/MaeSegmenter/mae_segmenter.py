import torch
from torch import nn
import torch.nn.functional as F
import timm

class MaeSegmenter(nn.Module):
    def __init__(self, encoder, decoder):
        super(MaeSegmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        patch_size = self.encoder.patch_embed.patch_size
        x = self.encoder.forward_features(x)    # N,3,H,W -> N,C,h,w
        x = self.decoder(x)                     # N,C,h,w -> N,n_cls,h,w
        x = F.interpolate(x, size=(H, W), mode='bilinear')
        return x
    
class LinearDecoder(nn.Module):
    def __init__(self, n_cls, embed_dim):
        super(LinearDecoder, self).__init__()
        self.n_cls = n_cls
        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, n_cls)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.head(x)
        return x.permute(0, 3, 1, 2)
    
def mae_segmenter(num_classes, pretrained, backbone='samvit_base_patch16', decoder_type='linear'):
    encoder = timm.create_model(backbone, pretrained=pretrained)
    decoder = LinearDecoder(num_classes, encoder.num_features)
    if decoder_type == 'deconv':
        raise NotImplementedError
    elif decoder_type == 'mask':
        raise NotImplementedError
    return MaeSegmenter(encoder, decoder)