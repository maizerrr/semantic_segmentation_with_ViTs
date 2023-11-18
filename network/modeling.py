from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
)
from .Segmenter import factory as segmenter
from .SwinDeepLab.swin_deeplab import SwinDeepLab
from .SwinDeepLab.configs.swin_224_7_4level import EncoderConfig, ASPPConfig, DecoderConfig
from .MaeSegmenter.mae_segmenter import mae_segmenter

import os
import torch

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split('_')[-1])
    inplanes = sum([hrnet_channels * 2 ** i for i in range(4)])
    low_level_planes = 256 # all hrnet version channel output from bottleneck is the same
    aspp_dilate = [12, 24, 36] # If follow paper trend, can put [24, 48, 72].

    if name=='deeplabv3plus':
        return_layers = {'stage4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'stage4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = xception.xception(pretrained= 'imagenet' if pretrained_backbone else False, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 128
    
    if name=='deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3', 'hrnetv2_32', output_stride, num_classes, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def segmenter_vit_base(num_classes=21, output_stride=None, pretrained_backbone=True):
    model_cfg = {
        'backbone': 'vit_base_patch8_384',
        'pretrained_backbone': pretrained_backbone,
        'd_model': 1024,
        'decoder': {
            'drop_path_rate': 0.0,
            'dropout': 0.1,
            'n_cls': num_classes,
            'n_layers': 1,
            'name': 'mask_transformer'
        },
        'drop_path_rate': 0.1,
        'dropout': 0.0,
        'image_size': (768,768),
        'n_cls': num_classes,
        'n_heads': 16,
        'n_layers': 24,
        'normalization': 'vit',
        'patch_size': 16
    }
    model = segmenter.create_segmenter(model_cfg)
    return model

# def segmenter_vit_deit(num_classes=21, output_stride=None, pretrained_backbone=True):
#     model_cfg = {
#         'backbone': 'vit_deit_base_distilled_patch16_384',
#         'pretrained_backbone': pretrained_backbone,
#         'd_model': 768,
#         'decoder': {
#             'drop_path_rate': 0.0,
#             'dropout': 0.1,
#             'n_cls': num_classes,
#             'n_layers': 1,
#             'name': 'mask_transformer'
#         },
#         'drop_path_rate': 0.1,
#         'dropout': 0.0,
#         'image_size': (768,768),
#         'n_cls': num_classes,
#         'n_heads': 12,
#         'n_layers': 12,
#         'normalization': 'vit',
#         'patch_size': 16
#     }
#     model = segmenter.create_segmenter(model_cfg)
#     return model

def swindeeplab_swin_t(num_classes=21, output_stride=8, pretrained_backbone=True):
    EncoderConfig.img_size = 768
    DecoderConfig.window_size = EncoderConfig.window_size = EncoderConfig.img_size // 32
    EncoderConfig.load_pretrained = pretrained_backbone
    DecoderConfig.num_classes = num_classes
    return SwinDeepLab(EncoderConfig, ASPPConfig, DecoderConfig)

def mae_segmenter_vit_base(num_classes=21, output_stride=8, pretrained_backbone=True):
    return mae_segmenter(num_classes=num_classes, pretrained=pretrained_backbone)