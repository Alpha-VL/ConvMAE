import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
try:
    import apex
except:
    print('apex is not installed')


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # use apex fp16 optimizer
    if cfg.optimizer_config.get("type", None) and cfg.optimizer_config["type"] == "DistOptimizerHook":
        if cfg.optimizer_config.get("use_fp16", False):
            model, optimizer = apex.amp.initialize(
                model.cuda(), optimizer, opt_level="O1")
            for m in model.modules():
                if hasattr(m, "fp16_enabled"):
                    m.fp16_enabled = True

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = 'IterBasedRunner' not in cfg.runner['type']
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
