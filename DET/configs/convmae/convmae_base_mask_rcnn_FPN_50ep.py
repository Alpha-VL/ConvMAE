from .convmae_base_mask_rcnn_FPN_100ep import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)

train.max_iter = train.max_iter // 2  # 100ep -> 50ep

lr_multiplier.warmup_length *= 2

train.output_dir = "./convmae_base_mask_rcnn_FPN_50ep"
__all__ = ["dataloader", "lr_multiplier", "model", "optimizer", "train"]
