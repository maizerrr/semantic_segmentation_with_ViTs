# Semantic Segmentation Models

Developed based on the following repos:
1. DeepLabV3/DeepLabV3+ PyTorch implementations https://github.com/VainF/DeepLabV3Plus-Pytorch
2. Segmenter PyTorch implementations https://github.com/rstrudel/segmenter
3. TransDeepLab PyTorch implementations https://github.com/rezazad68/transdeeplab

## Quick Start 

### 1. Available Architectures
|DeepLabV3+|TransDeepLab|Segmenter(Mask)|MaeSegmenter(Linear)|MaeSegmenter(Deconv)|MaeSegmenter(Mask)
|:---:|:---:|:---:|:---:|:---:|:---:|
|deeplabv3plus_resnet50|swindeeplab_resnet50|-|-|-|-|
|deeplabv3plus_resnet101|-|-|-|-|-|
|deeplabv3plus_mobilenet|-|-|-|-|-|
|deeplabv3plus_xception|swindeeplab_xception|-|-|-|-|
|-|-|swindeeplab_swin_t|-|-|-|-|
|-|-|-|-|mae_semgemter_vit_base|mae_semgemter_vit_base_deconv|mae_semgemter_vit_base_mask|
|-|-|-|segmenter_vit_large|-|-|-|
|-|-|-|-|mae_semgemter_vit_huge|mae_semgemter_vit_huge_deconv|mae_semgemter_vit_huge_mask|

please refer to [network/modeling.py](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/modeling.py) for all model entries.

### 2. Load the pretrained model:
```python
model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_SRTIDE)
model.load_state_dict( torch.load( PATH_TO_PTH )['model_state']  )
```
### 3. Visualize segmentation outputs:
```python
outputs = model(images)
preds = outputs.max(1)[1].detach().cpu().numpy()
colorized_preds = val_dst.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# Do whatever you like here with the colorized segmentation maps
colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
```

### 4. Atrous Separable Convolution

**Note**: All pre-trained models in this repo were trained without atrous separable convolution.

Atrous Separable Convolution is supported in this repo. We provide a simple tool ``network.convert_to_separable_conv`` to convert ``nn.Conv2d`` to ``AtrousSeparableConvolution``. **Please run main.py with '--separable_conv' if it is required**. See 'main.py' and 'network/_deeplab.py' for more details. 

### 5. Prediction
Single image:
```bash
python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen/bremen_000000_000019_leftImg8bit.png  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```

Image folder:
```bash
python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results
```

### 6. New datasets

You can train deeplab models on your own datasets. Your ``torch.utils.data.Dataset`` should provide a decoding method that transforms your predictions to colorized images, just like the [VOC Dataset](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/bfe01d5fca5b6bb648e162d522eed1a9a8b324cb/datasets/voc.py#L156):
```python

class MyDataset(data.Dataset):
    ...
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
```


## Results

### 1. Performance on Pascal VOC2012 Aug (21 classes, 513 x 513)
N/A

### 2. Performance on Cityscapes (19 classes, 1024 x 2048)

Training: 768x768 random crop  
validation: 1024x2048

| Model | Oputput Stride/Patch Size | params | FPS | mIoU |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| DeepLabV3Plus-MobileNet | 16 | 5.23M | ? | ? |
| DeepLabV3Plus-Xception | 16 | 37.05M | 45 | ? |
| DeepLabV3Plus-ResNet101 | 16 | 58.75M | 15 | 76.1% |
| TransDeepLab-SwinT | 16 | ? | ? | ? |
| Segmenter-ViT_Large | 16 | 322.25M | 5 | 76.1% |
| MaeSegmenter-ViT_Base | 16 | 89.68M | 3 | 75.3% |
| MaeSegmenter-ViT_Huge | 16 | 631.03M | ? | ? |
| MaeSegmenter-ViT_Base-DeConv | 16 | ? | ? | ? |
| MaeSegmenter-ViT_Base-Mask | 16 | ? | ? | ? |

p.s. FPS is measured based on the time required to process a single sample (1x3x1024x2048) during a forward pass

## Pascal VOC

N/A

## Cityscapes

### 1. Download cityscapes and extract it to 'datasets/data/cityscapes'

```
/datasets
    /data
        /cityscapes
            /gtFine
            /leftImg8bit
```

### 2. Train your model on Cityscapes

**Option 1: use command line**

```bash
python main.py --model mae_segmenter_vit_base --dataset cityscapes --gpu_id 0 --total_epochs 100 --base_lr 0.1 --loss_type focal_loss --crop_size 768 --batch_size 1 --val_batch_size 1 --use_amp --output_stride 16 --data_root ./datasets/data/cityscapes 
```

**Option 2: use WebUI**

Open main_gui.ipynb in Jupyter Notebook and follow the instructions

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
