<div align="center">
<h1>ConvMAE</h1>
<h3>ConvMAE: Masked Convolution Meets Masked Autoencoders</h3>

[Peng Gao](https://scholar.google.com/citations?user=miFIAFMAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Teli Ma](https://scholar.google.com/citations?user=arny77IAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Hongsheng Li](https://scholar.google.com/citations?user=BN2Ze-QAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Jifeng Dai](https://scholar.google.com/citations?user=SH_-B_AAAAAJ&hl=en&oi=ao)<sup>3</sup>, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en&oi=ao)<sup>1</sup>,

<sup>1</sup> [Shanghai AI Laboratory](https://www.shlab.org.cn/), <sup>2</sup> [MMLab, CUHK](https://mmlab.ie.cuhk.edu.hk/), <sup>3</sup> [Sensetime Research](https://www.sensetime.com/cn).

</div>

This repo is the official implementation of [ConvMAE: Masked Convolution Meets Masked Autoencoders](). It currently concludes codes and models for the following tasks:
> **ImageNet Pretrain**: See [PRETRAIN.md](PRETRAIN.md).\
> **ImageNet Finetune**: See [FINETUNE.md](FINETUNE.md).\
> **Object Detection**: See [DETECTION.md](DETECTION.md).\
> **Semantic Segmentation**: See [SEGMENTATION.md](SEGMENTATION.md).

## Introduction
ConvMAE framework demonstrates that multi-scale hybrid convolution-transformer can learn more discriminative representations via the mask auto-encoding scheme. 
* We present the strong and efficient self-supervised framework ConvMAE, which is easy to implement but show outstanding performances on downstream tasks.
* ConvMAE naturally generates hierarchical representations and exhibit promising performances on object detection and segmentation.
<!-- * ConvMAE-Base improves the ImageNet finetuning accuracy by 1.3% compared with MAE-Base.
On object detection with Mask-RCNN, ConvMAE-Base achieves 52.5 box AP and 46.5 mask AP with a 25-epoch training schedule while MAE-Base attains 50.3 box AP and 44.9 mask AP with 100 training epochs. On ADE20K with UperNet, ConvMAE-Base surpasses MAE-Base by 2.6 mIoU (48.1 vs. 50.7). -->


![tenser](figures/ConvMAE.png)


## Main Results on ImageNet-1K
| Models | #Params(M) | GFLOPs |Pretrain Epochs | Finetune acc@1(%) | Linear Probe acc@1(%) | logs | weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ConvMAE-B | 88 | |1600 |  | |  |  | 
| ConvMAE-L | 322 | | 800 | | | | |


## Main Results on COCO
| Models | Pretrain Epochs | Finetune Epochs | #Params(M)| GFLOPs | box AP | mask AP | logs | weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ConvMAE-B |  | | |  | |  |  | 

## Main Results on ADE20K
| Models |Pretrain Epochs| Finetune Iters | #Params(M)| GFLOPs | mIoU | logs | weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ConvMAE-B |  | | |  | |  |  | 

## Usage
### Prerequisites
* Linux
* Python 3.7+
* CUDA 10.2+
* GCC 5+

### Training and inference
* See [PRETRAIN.md](PRETRAIN.md) for pretraining.
* See [FINETUNE.md](FINETUNE.md) for pretrained model finetuning and linear probing. 
* See [DETECTION.md](DETECTION.md) for using pretrained backbone on [Mask RCNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html).
* See [SEGMENTATION.md](SEGMENTATION.md) for using pretrained backbone on [UperNet](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html).

## Acknowledgement
The pretraining and finetuning of our project are based on [DeiT](https://github.com/facebookresearch/deit) and [MAE](https://github.com/facebookresearch/mae). The object detection and semantic segmentation parts are based on [MIMDet](https://github.com/hustvl/MIMDet) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) respectively. Thanks for their wonderful work.

## License
ConvMAE is released under the [MIT License](https://github.com/Alpha-VL/ConvMAE/blob/main/LICENSE).

## Citation

