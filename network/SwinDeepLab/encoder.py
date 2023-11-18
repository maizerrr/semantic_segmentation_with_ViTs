import os, ssl
import urllib.request
import timm
import torch
from torch import nn
from .backbones import resnets
from .backbones.swin import SwinEncoder
from .backbones.xception import AlignedXception
from .backbones.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from ..utils import download_pretrained

def build_encoder(config):
    if config.encoder_name == 'swin':
        if config.norm_layer == 'layer':
            norm_layer = nn.LayerNorm
            
        encoder = SwinEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            high_level_idx=config.high_level_idx,
            low_level_idx=config.low_level_idx,
            high_level_after_block=config.high_level_after_block,
            low_level_after_block=config.low_level_after_block,
            embed_dim=config.embed_dim, 
            depths=config.depths, 
            num_heads=config.num_heads,
            window_size=config.window_size, 
            mlp_ratio=config.mlp_ratio, 
            qkv_bias=config.qkv_bias, 
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate, 
            attn_drop_rate=config.attn_drop_rate, 
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer,
            high_level_norm=config.high_level_norm,
            low_level_norm=config.low_level_norm,
            ape=config.ape, 
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint
        )

        if config.load_pretrained:
            path = torch.hub.get_dir() + "/checkpoints/"
            file = "swin_tiny_patch4_window7_224.pth"
            url = "https://github.com/maizerrr/semantic_segmentation_with_ViTs/releases/download/untagged-1f7e5dd85d0510f02701/swin_tiny_patch4_window7_224.pth"
            download_pretrained(url, path, filename=file)
            encoder.load_from(path+file)

        return encoder
        
    if config.encoder_name == 'xception':
        if config.sync_bn:
            bn = SynchronizedBatchNorm2d
        else:
            bn = nn.BatchNorm2d
        return AlignedXception(output_stride=config.output_stride,
                               input_size=config.img_size,
                               BatchNorm=bn, pretrained=config.pretrained,
                               high_level_dim=config.high_level_dim)
    
    if config.encoder_name == 'resnet':
        model = timm.create_model('resnet50_encoder', 
                                  pretrained=False,
                                  high_level=None,
                                  num_classes=0)
        if config.load_pretrained:
            path = torch.hub.get_dir() + "/checkpoints/"
            file = "resnet50_a1_0-14fe96d1.pth"
            url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth"
            download_pretrained(url, path, filename=file)
                
            weight = torch.load(path+file)
            msg = model.load_state_dict(weight, strict=False)
            print(msg)
        
        model.layer4 = nn.Identity()
        model.high_level_size = 14
        model.high_level_dim = 384
        model.low_level_dim = 128
        
        return model