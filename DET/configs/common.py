from typing import Dict, List, Optional
from omegaconf import DictConfig

import torch
from detectron2.config import LazyCall as L
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.solver import WarmupParamScheduler
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from fvcore.common.param_scheduler import MultiStepParamScheduler

from models.modeling import _postprocess


class GeneralizedRCNNImageListForward(GeneralizedRCNN):
    def __init__(self, *args, **kwargs):
        self.lsj_postprocess = kwargs.pop("lsj_postprocess")
        super().__init__(*args, **kwargs)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            if self.lsj_postprocess:
                return _postprocess(results, batched_inputs, images.image_sizes)
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results


def get_fpn_model_parameters(
    model,
    weight_decay=1e-5,
    weight_decay_norm=0.0,
    base_lr=4e-5,
    skip_list=(),
    multiplier=1.5,
):
    parameter_group_vars = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        elif "norm" in name and weight_decay_norm is not None:
            group_name = "decay"
            this_weight_decay = weight_decay_norm
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if name.startswith("backbone.bottom_up.encoder.patch_embed"):
            group_name = "backbone.bottom_up.encoder.patch_embed_%s" % (group_name)
            if group_name not in parameter_group_vars:
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": base_lr,
                }
        elif name.startswith("backbone.bottom_up.encoder"):
            group_name = "backbone.bottom_up.encoder_%s" % (group_name)
            if group_name not in parameter_group_vars:
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": base_lr / multiplier,
                }
        else:
            group_name = "others_%s" % (group_name)
            if group_name not in parameter_group_vars:
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": base_lr * multiplier,
                }

        parameter_group_vars[group_name]["params"].append(param)
    return list(parameter_group_vars.values())


train = dict(
    output_dir="",
    init_checkpoint="",
    max_iter=368750,
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False, find_unused_parameters=False, fp16_compression=True,
    ),
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_period=20,
    device="cuda"
    # ...
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[327778, 355029],
        num_updates=train["max_iter"],
    ),
    warmup_length=500 / train["max_iter"],
    warmup_factor=0.067,
)

