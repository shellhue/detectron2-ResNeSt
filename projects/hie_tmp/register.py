import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

_HIE_DATASETS = {
    # 数据集名字 : (对应的图片目录， 对应的json文件)
    "pretrain_and_hie_train": ("pretrain_and_hie_coco/train", "pretrain_and_hie_coco/annotations/instances_train.json"),
    "pretrain_and_hie_test": ("pretrain_and_hie_coco/test", "pretrain_and_hie_coco/annotations/instances_test.json"),
    "hie_train": ("hie_coco_all/train", "hie_coco_all/annotations/instances_train.json"),
    "hie_val": ("hie_coco_all/val", "hie_coco_all/annotations/instances_val.json"),
    "hie_test": ("hie_coco_all/test", "hie_coco_all/annotations/instances_test.json"),
    "hie_part_train": ("hie_coco_part/train", "hie_coco_part/annotations/instances_train.json"),
    "hie_part_test": ("hie_coco_all/test", "hie_coco_all/annotations/instances_test.json"),
    "hie_part_val": ("hie_coco_part/val", "hie_coco_part/annotations/instances_val.json"),
}


def _get_hie_metadata():
    return {
        "thing_dataset_id_to_contiguous_id": {
            1: 0
        },
        "thing_classes": ["person"],
        "thing_colors": [[220, 20, 60]],
    }

def register_hie(root):
    for key, (image_root, json_file) in _HIE_DATASETS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_hie_metadata(),
            # _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

register_hie("/home/huangzeyu/tmp/detectron2-ResNeSt/projects/hie_tmp")