import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_

from ..Segmenter.blocks import Block
from ..Segmenter.util import init_weights

class MaeSegmenter(nn.Module):
    def __init__(self, encoder, decoder):
        super(MaeSegmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)    # N,3,H,W -> N,C,h,w
        x = self.decoder(x)    # N,C,h,w -> N,n_cls,h,w
        return x
    
class LinearDecoder(nn.Module):
    def __init__(self, n_cls, embed_dim, patch_size):
        super(LinearDecoder, self).__init__()
        self.patch_size = patch_size
        self.head = nn.Linear(embed_dim, n_cls)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.head(x)
        x = x.permute(0, 3, 1, 2)
        return F.interpolate(x, scale_factor=self.patch_size, mode='bilinear')
    
class DeconvDecoder(nn.Module):
    def __init__(self, n_cls, embed_dim, patch_size):
        super(DeconvDecoder, self).__init__()
        # self.head = nn.ConvTranspose2d(
        #     in_channels=embed_dim, 
        #     out_channels=n_cls, 
        #     kernel_size=patch_size, 
        #     stride=patch_size)
        self.fc1 = nn.Linear(embed_dim, n_cls)
        self.fc2 = nn.ConvTranspose2d(in_channels=n_cls, out_channels=n_cls, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.fc2(x)
        return x

class MaskDecoder(nn.Module):
    def __init__(self, n_cls, embed_dim, patch_size, n_layers=1, drop_path_rate=0.0, dropout=0.1):
        super(MaskDecoder, self).__init__()
        self.n_cls = n_cls
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(embed_dim, embed_dim//64, embed_dim*4, dropout, dpr[i]) for i in range(n_layers)]
        )
        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, embed_dim))
        self.proj_patch = nn.Parameter(embed_dim ** -0.5 * torch.randn(embed_dim, embed_dim))
        self.proj_classes = nn.Parameter(embed_dim ** -0.5 * torch.randn(embed_dim, embed_dim))
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.mask_norm = nn.LayerNorm(n_cls)
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        x = torch.reshape(x.permute(0, 2, 3, 1), (-1, H*W, self.embed_dim))
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:,:-self.n_cls], x[:,-self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1,2)
        masks = self.mask_norm(masks)
        masks = torch.reshape(masks.permute(0, 2, 1), (-1, self.n_cls, H, W))
        return F.interpolate(masks, scale_factor=self.patch_size, mode='bilinear')

    
def mae_segmenter(num_classes, pretrained, backbone='samvit_base_patch16', decoder_type='linear'):
    encoder = timm.create_model(backbone, pretrained=pretrained)
    encoder.forward = encoder.forward_features
    decoder = LinearDecoder(num_classes, encoder.num_features, encoder.patch_embed.patch_size)
    if decoder_type == 'deconv':
        decoder = DeconvDecoder(num_classes, encoder.num_features, encoder.patch_embed.patch_size)
    elif decoder_type == 'mask':
        encoder.neck = nn.Identity()
        decoder = MaskDecoder(num_classes, encoder.embed_dim, encoder.patch_embed.patch_size)
    return MaeSegmenter(encoder, decoder)