# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from yacs.config import CfgNode
import os
import torch.nn.functional as F
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import datasets.DAOD as DAOD
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from config import get_cfg_defaults
from datasets.data_prefetcher import data_prefetcher


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def setup(args):
    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    utils.init_distributed_mode(cfg)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        os.system(f'cp {args.config_file} {cfg.OUTPUT_DIR}')
        ddetr_src = 'models/deformable_detr.py'
        ddetr_des = Path(cfg.OUTPUT_DIR) / 'deformable_detr.py.backup'
        dtrans_src = 'models/deformable_transformer.py'
        dtrans_des = Path(cfg.OUTPUT_DIR) / 'deformable_transformer.py.backup'
        main_src = 'main.py'
        main_des = Path(cfg.OUTPUT_DIR) / 'main.py.backup'
        os.system(f'cp {ddetr_src} {ddetr_des}')
        os.system(f'cp {dtrans_src} {dtrans_des}')
        os.system(f'cp {main_src} {main_des}')

    return cfg

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_map = None
        self.gradients = None

        # 注册 Hook
        self.hook_feature_map()
        self.hook_gradients()

    def hook_feature_map(self):
        def hook_fn(module, input, output):
            self.feature_map = output.detach()

        # 假设最后一个卷积层是 model.backbone 或者相应的层
        self.model.backbone[-1].register_forward_hook(hook_fn)

    def hook_gradients(self):
        def hook_fn(module, input, output):
            output.register_hook(lambda grad: self.gradients.append(grad))

        self.model.backbone[-1].register_backward_hook(hook_fn)

    def generate_cam(self, class_idx):
        # 计算权重
        weights = self.gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU
        cam = cam - cam.min()
        cam = cam / cam.max()  # 归一化
        return cam.squeeze().cpu().numpy()

def main(cfg):
    align = cfg.MODEL.BACKBONE_ALIGN or cfg.MODEL.SPACE_ALIGN or cfg.MODEL.CHANNEL_ALIGN or cfg.MODEL.INSTANCE_ALIGN
    assert align == (cfg.DATASET.DA_MODE == 'uda')

    print("git:\n  {}\n".format(utils.get_sha()))
    print(cfg)
    tb_writer = SummaryWriter()
    if cfg.MODEL.FROZEN_WEIGHTS is not None:
        assert cfg.MODEL.MASKS, "Frozen training is meant for segmentation only"
    import os

    # cuda_visible_devices = os.getenv(cfg.ENVIRONMENT.CUDA_VISIBLE_DEVICES)
    # print(f"Using GPU: {cuda_visible_devices}")

    # device = torch.device(cfg.DEVICE)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.cuda.set_device(device.index)

    # fix the seed for reproducibility
    seed = cfg.SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', cfg=cfg)
    dataset_val = build_dataset(image_set='val', cfg=cfg)

    if cfg.DIST.DISTRIBUTED:
        if cfg.CACHE_MODE:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if cfg.DATASET.DA_MODE == 'uda':
        assert cfg.TRAIN.BATCH_SIZE % 2 == 0, f'cfg.TRAIN.BATCH_SIZE {cfg.TRAIN.BATCH_SIZE} should be a multiple of 2'
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.TRAIN.BATCH_SIZE//2, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=DAOD.collate_fn, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)
    else:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.TRAIN.BATCH_SIZE, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)
    data_loader_val = DataLoader(dataset_val, cfg.TRAIN.BATCH_SIZE*5, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        # print(n)
        pass

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and not match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.TRAIN.LR,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and p.requires_grad],
            "lr": cfg.TRAIN.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.LR_LINEAR_PROJ_MULT,
        }
    ]
    if cfg.TRAIN.SGD:
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.LR, momentum=0.9,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP)

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', patience=5, factor=0.3, threshold=0.001, verbose=True)

    if cfg.DIST.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DIST.GPU])
        model_without_ddp = model.module

    if cfg.DATASET.DATASET_FILE == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", cfg)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if cfg.MODEL.FROZEN_WEIGHTS is not None:
        checkpoint = torch.load(cfg.MODEL.FROZEN_WEIGHTS, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(cfg.OUTPUT_DIR)
    if cfg.RESUME: # [BUG] write after freezing cfgs
        if cfg.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.RESUME, map_location='cpu')
            print("----------resume:",cfg.RESUME)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        # missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not cfg.EVAL and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            override_resumed_lr_drop = True
            if override_resumed_lr_drop:
                print('Warning: (hack) override_resumed_lr_drop is set to True, so cfg.TRAIN.LR_DROP would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = cfg.TRAIN.LR_DROP
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            cfg = CfgNode(cfg, new_allowed=True)
            cfg.START_EPOCH = checkpoint['epoch'] + 1 
        # check the resumed model
        if not cfg.EVAL:
            test_stats, coco_evaluator,map50 = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
            )
    
    if cfg.EVAL:
        test_stats, coco_evaluator,map50 = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, cfg.OUTPUT_DIR)
        if cfg.OUTPUT_DIR:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    best_map50 = 0.15
##############################################
    # Build MAE branch
    prefetcher = data_prefetcher(data_loader_train, device, prefetch=True)
    samples, targets = prefetcher.next()
    samples_size = samples.tensors.shape[-2:]
    print("size:",samples_size)
    image_size = data_loader_train.dataset.__getitem__(0)[0].shape[-2:]
    print("训练集尺寸0：源域",data_loader_train.dataset.__getitem__(0)[0].shape[-2:])
    print("训练集尺寸1目标域：",data_loader_train.dataset.__getitem__(0)[1].shape[-2:])
    # image_size = [530,1000]
    # model.transformer.build_mae_decoder(samples_size, cfg.mae_layers, device, channel0=model.backbone.num_channels[0])







#############################################################


    #########################################################
#########################################################
    for epoch in range(cfg.START_EPOCH, cfg.TRAIN.EPOCHS):
        if cfg.DIST.DISTRIBUTED:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, mask_ratio=cfg.mask_ratio,optimizer = optimizer, device = device, epoch = epoch, max_norm=cfg.TRAIN.CLIP_MAX_NORM,)
        lr_scheduler.step()
        if cfg.OUTPUT_DIR:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % cfg.TRAIN.LR_DROP == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': cfg,
                }, checkpoint_path)

        test_stats, coco_evaluator,map50 = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
        )

        if map50 > best_map50:
            best_map50 = map50

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
            }, 'exps/r50_uda_c2fc/best_map50_checkpoint.pth')
            print('\n\n----Best MAP50:{:.3f}---\n\n'.format(best_map50))

        # 记录tensorboard
        # lr_scheduler.step(test_stats["loss"])

        tb_writer.add_scalar('MAP_50', map50, epoch)
        tb_writer.add_scalars("loss", {
            "train_loss": train_stats["loss"],
            "test_loss": test_stats["loss"]
        }, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if cfg.OUTPUT_DIR and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector')
    parser.add_argument('--config_file', default='configs/r50_uda_c2fc.yaml', type=str)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup(args)
    main(cfg)

