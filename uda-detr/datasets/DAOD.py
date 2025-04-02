# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


from pathlib import Path
from torch.utils.data import Dataset

from datasets.coco import CocoDetection, make_coco_transforms
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list


def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': 'D:\ZTT\domainData\cityscapes\leftImg8bit/train',
            # 'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'train_anno': root / 'D:\ZTT\domainData\cityscapes\coco_annotations/cityscapes_train_cocostyle.json',
            # 'val_img': 'cityscapes/leftImg8bit/val',
            'val_img': 'D:\ZTT\domainData\cityscapes\leftImg8bit/val',
            # 'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
            'val_anno':  'D:\ZTT\domainData\cityscapes\coco_annotations/cityscapes_val_cocostyle.json',
        },
        'cityscapes_caronly': {
            'train_img': 'D:\ZTT\domainData\cityscapes\leftImg8bit/train',
            'train_anno': 'D:\ZTT\domainData\cityscapes\coco_annotations/cityscapes_train_caronly_cocostyle.json',
            'val_img': 'D:\ZTT\domainData\cityscapes\leftImg8bit/val',
            'val_anno': 'D:\ZTT\domainData\cityscapes\coco_annotations/cityscapes_val_caronly_cocostyle.json',
        },
        'foggy_cityscapes': {
            'train_img': 'D:\ZTT\domainData\cityscapes_foggy\leftImg8bit_foggy/train',
            'train_anno': 'D:\ZTT\domainData\cityscapes_foggy\coco_annotations/foggy_cityscapes_train_cocostyle.json',
            'val_img': 'D:\ZTT\domainData\cityscapes_foggy\leftImg8bit_foggy/val',
            'val_anno': 'D:\ZTT\domainData\cityscapes_foggy\coco_annotations/foggy_cityscapes_val_cocostyle.json',
        },
        'sim10k': {
            'train_img': 'D:\ZTT\domainData\sim10k\JPEGImages',
            'train_anno': 'D:\ZTT\domainData\sim10k/coco_annotations/sim10k_MRT_cocostyle.json',
        },
        'bdd_daytime': {
            # 'train_img': root / 'bdd_daytime/train',
            'train_img':  'D:\ZTT\domainData\BDD100K/bdd100k\images/100k/train',
            'train_anno': 'D:\ZTT\domainData\BDD100K/annotations_coco/bdd_daytime_train.json',
            # 'val_img': root / 'bdd_daytime/val',
            'val_img':  'D:\ZTT\domainData\BDD100K/bdd100k\images/100k/val',
            'val_anno':  'D:\ZTT\domainData\BDD100K/annotations_coco/bdd_daytime_val.json',
        },
        'kitti': {
            'train_img': 'D:/ZTT/domainData/KITTI/object/training/image_2',
            'train_anno': 'D:\ZTT\domainData\KITTI\object/training\label\kitti_caronly_coco/coco_format.json',
        },
    }


class DADataset(Dataset):
    def __init__(self, source_img_folder, source_ann_file, target_img_folder, target_ann_file,
                 transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        self.source = CocoDetection(
            img_folder=source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        self.target = CocoDetection(
            img_folder=target_img_folder,
            ann_file=target_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

    def __len__(self):
        # return max(len(self.source), len(self.target))
        return len(self.source)

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        target_img, _ = self.target[idx % len(self.target)]
        return source_img, target_img, source_target


def collate_fn(batch):
    source_imgs, target_imgs, source_targets = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(source_imgs + target_imgs)
    return samples, source_targets


def build(image_set, cfg):
    paths = get_paths(cfg.DATASET.COCO_PATH)
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_')
    if image_set == 'val':
        return CocoDetection(
            img_folder=paths[target_domain]['val_img'],
            ann_file=paths[target_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )
    elif image_set == 'train':
        if cfg.DATASET.DA_MODE == 'source_only':
            return CocoDetection(
                img_folder=paths[source_domain]['train_img'],
                ann_file=paths[source_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'oracle':
            return CocoDetection(
                img_folder=paths[target_domain]['train_img'],
                ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'uda':
            return DADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')
