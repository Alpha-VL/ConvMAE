# ConvMAE: Masked Convolution Meets Masked Autoencoders

This folder contains the implementation of the ConvMAE finetuning for image classification.

## Model Zoo

| Models | #Params(M) | Supervision | Encoder Ratio | Pretrain Epochs | FT acc@1(%) | FT logs/weights |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ConvMAE-B | 88 | RGB | 25% | 1600 | 85.0 | [log](https://drive.google.com/file/d/1nzAOD5UR3b9QqwD2vMMz0Bx3170sypuy/view?usp=sharing)/[weight](https://drive.google.com/file/d/19F6vQUlITpzNLvXLKi5NRxRLOmKRxqFi/view?usp=sharing) |

## Usage

### Install
- Clone this repo:

```bash
git clone https://github.com/Alpha-VL/ConvMAE
cd ConvMAE
```

- Create a conda environment and activate it:
```bash
conda create -n convmae python=3.7
conda activate convmae
```

- Install `Pytorch==1.8.0` and `torchvision==0.9.0` with `CUDA==11.1`

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Install `timm==0.3.2`

```bash
pip install timm==0.3.2
```

### Data preparation

You can download the ImageNet-1K [here](https://image-net.org) and prepare the ImageNet-1K follow this format:

```tree data
imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
```
### Evaluation

Download the finetuned model from [here](https://drive.google.com/file/d/19F6vQUlITpzNLvXLKi5NRxRLOmKRxqFi/view?usp=sharing).

Evaluate ConvViT-Base by running:

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py --batch_size 128 --model convvit_base_patch16 --resume ${FINETUNE_CHKPT} --dist_eval --data_path ${IMAGENET_DIR} --eval
``` 

This shoud give:

```bash
* Acc@1 84.982 Acc@5 97.152 loss 0.695
Accuracy of the network on the 50000 test images: 85.0%
```

### Fine-tuning
Download the pretrained model from [here](https://drive.google.com/file/d/1AEPivXw0A0b_m5EwEi6fg2pOAoDr8C31/view?usp=sharing).

To finetune with multi-node distributed training, run the following on 4 nodes with 8 GPUs each:
```bash
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 32 \
    --model convvit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

To finetune with single-node training, run the following on single node with 8 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --batch_size 128 \
    --model convvit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
#### Notes
- There are chances that loss is nan during finetuning process, if so, just delete the [line](https://github.com/Alpha-VL/ConvMAE/blob/53d56ad2388665bf86e0e029aa3f424e709a6652/engine_finetune.py#L55) to use fp32 type to resume the finetuning from where it broke down.
- How to resume: just add `--resume` into above scripts as:
```bash
--resume ${CHKPT_RESUME}
```
- Also, we are still working to solve the possible gradient vanish caused by fp16 mixed-precision finetuning. Feeling free to contact us if you have any suggestions.

### Linear Probing
Download the pretrained model from [here](https://drive.google.com/file/d/1AEPivXw0A0b_m5EwEi6fg2pOAoDr8C31/view?usp=sharing).

To finetune with multi-node distributed training, run the following on 4 nodes with 8 GPUs each:
```bash
python submitit_linprobe.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 128 \
    --model convvit_base_patch16 \
    --global_pool \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

To finetune with single-node training, run the following on single node with 8 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --batch_size 512 \
    --model convvit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

