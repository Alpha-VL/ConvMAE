import detectron2.data.transforms as T
import torch
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    StandardROIHeads,
)
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from fvcore.common.param_scheduler import CosineParamScheduler

from models import ConvViTDet

from ..coco import dataloader
from ..common import GeneralizedRCNNImageListForward

model = L(GeneralizedRCNNImageListForward)(
    lsj_postprocess=True,
    backbone=L(FPN)(
        bottom_up=L(ConvViTDet)(
            window_size=16,
            with_cp=False,
            pretrained="./convmae_base.pth",
            stop_grad_conv1=False,
            sincos_pos_embed=True,
            zero_pos_embed=False,
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            init_values=None,
            beit_qkv_bias=False,
        ),
        in_features=["s0", "s1", "s2", "s3"],
        out_channels=256,
        norm="SyncBN",
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.12, 57.375],
    input_format="RGB",
)
# Using NaiveSyncBatchNorm because heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
model.roi_heads.box_head.conv_norm = (
    model.roi_heads.mask_head.conv_norm
) = lambda c: NaiveSyncBatchNorm(c, stats_mode="N")
# fmt: on

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        overrides={
            "pos_embed": {"weight_decay": 0.0},
            "relative_position_bias_table": {"weight_decay": 0.0},
        },
    ),
    lr=8.0e-05,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(start_value=1.0, end_value=0.0),
    warmup_length=0.125 / 100,
    warmup_factor=0.001,
)

train = dict(
    output_dir="output/convmae_base_mask_rcnn_FPN_100ep",
    init_checkpoint="",
    max_iter=368750,
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False, find_unused_parameters=False, fp16_compression=True,
    ),
    checkpointer=dict(period=3688, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=3688,
    log_period=20,
    device="cuda"
    # ...
)

# resize_and_crop_image in:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/utils/input_utils.py#L127  # noqa: E501, B950
image_size = 1024
dataloader.train.total_batch_size = 32
dataloader.train.mapper.augmentations = [
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
    L(T.RandomFlip)(horizontal=True),
]
dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.image_format = "RGB"
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True
dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size)),
]
dataloader.test.mapper.image_format = "RGB"
dataloader.evaluator.output_dir = "${...train.output_dir}"
