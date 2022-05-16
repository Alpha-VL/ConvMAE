from .convmae_base_mask_rcnn_FPN_100ep import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter = train.max_iter // 4  # 100ep -> 25ep

lr_multiplier.warmup_length *= 4

train.output_dir = "./convmae_base_mask_rcnn_FPN_25ep"
__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]
