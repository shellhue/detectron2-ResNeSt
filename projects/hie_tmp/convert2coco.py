import os
from os import path
import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def get_subfolders_recursively(folder_path):
    """Get all subfolders recursively in folder_path.
    """
    folder_list = []
    for root, dirs, _ in os.walk(folder_path):
        for one_dir in dirs:
            one_dir = os.path.join(root, one_dir)
            folder_list.append(one_dir)
    return folder_list


def _get_direct_files_in_dir(dir_path, formats):
    """Get all direct imgs in dir_path.
    """
    imgs = []
    files = os.listdir(dir_path)
    for file in files:
        f = os.path.splitext(file)[1]
        if f in formats:
            imgs.append(os.path.join(dir_path, file))
    return imgs


def get_all_imgs_in_dir(root_dir):
    """Get all imgs recursively in dir.
    """
    all_imgs = []
    subfolders = get_subfolders_recursively(root_dir)
    subfolders.append(root_dir)
    for folder in subfolders:
        imgs = _get_direct_files_in_dir(
            folder, [".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"])
        all_imgs.extend(imgs)
    return all_imgs

def get_coco_infos(imgs, output_dir, copy_img=False):
    imgs_info = []
    annotations = []
    for i, x in enumerate(tqdm(imgs)):
        l = x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        img = cv2.imread(x)
        height, width, _ = img.shape
        img_name = '{0:08d}.jpg'.format(i)
        img_o = path.join(output_dir, img_name)
        if copy_img:
            cv2.imwrite(img_o, img)
            # shutil.copy(x,img_o)
        imgs_info.append({
            "file_name": img_name,
            "width": width,
            "height": height,
            "id": i
        })
        if os.path.exists(l):
            labels = np.loadtxt(l, dtype=np.float32).reshape(-1, 6)
            labels = labels[:, 2:6]
            
            labels[:, [0,2]] = labels[:, [0,2]] * width
            labels[:, [1,3]] = labels[:, [1,3]] * height
            labels[:, 0:2] = labels[:, 0:2] - labels[:, 2:4] / 2.0
            for l in labels.tolist():
                annotations.append({
                    "area": l[2] * l[3],
                    "iscrowd": 0,
                    "image_id": i,
                    "bbox": l,
                    "category_id": 1,
                    "id": len(annotations)
                })
    return imgs_info, annotations

def hie2coco(videos, root="/fs/disk1/MOTDataset/HIE20", src_img_dir="images/train", output_dir="./hie_coco", ds_type="train"):
    anns_dir = path.join(output_dir, "annotations")
    imgs_dir = path.join(output_dir, ds_type)
    
    for d in [anns_dir, imgs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    imgs = []
    
    for v in videos:
        video_dir = path.join(path.join(root, src_img_dir), v)
        imgs.extend(get_all_imgs_in_dir(video_dir))
    imgs_info, annotations = get_coco_infos(imgs, imgs_dir, copy_img=True)
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]
    data = {
        "images": imgs_info,
        "annotations": annotations,
        "categories": categories,
    }
    with open(path.join(anns_dir, "instances_{}.json".format(ds_type)), 'w') as f:
        json.dump(data, f)

def get_imgs_from_data_cfg(data_cfg="/home/huangzeyu/FairMOT/src/lib/cfg/datanomot20.json"):
    data_cfg = json.load(open(data_cfg))
    ds_root = data_cfg["root"]
    train_files = data_cfg["train"].values()

    imgs = []
    for n in train_files:
        with open(n) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            content = [os.path.join(ds_root, x) for x in content]
            imgs.extend(content)
    return imgs   

def get_imgs_from_hie_video_list(videos, hie_dataset_root="/fs/disk1/MOTDataset/HIE20"):
    imgs = []
    for v in videos:
        img_dir = "{}/images/train/{}/img".format(hie_dataset_root, v)
        imgs.extend(get_all_imgs_in_dir(img_dir))
    return imgs

def imgs2coco(imgs, output_dataset_path="./hie_coco", ds_type="train"):
    anns_dir = path.join(output_dataset_path, "annotations")
    imgs_dir = path.join(output_dataset_path, ds_type)
    
    for d in [anns_dir, imgs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    imgs_info, annotations = get_coco_infos(imgs, imgs_dir, copy_img=True)
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]
    data = {
        "images": imgs_info,
        "annotations": annotations,
        "categories": categories,
    }
    with open(path.join(anns_dir, "instances_{}.json".format(ds_type)), 'w') as f:
        json.dump(data, f)
    
    
if __name__ == "__main__":
    
    # 1. 生成datanomot20+hie的pretrain coco格式数据集
    all_videos=["2", "4", "6", "7", "8", "10", "11", "13", "14", "15", "16", "18", "1", "3", "5", "9", "12", "17", "19"]

    all_imgs = []
    
    imgs_from_nohie_datasets = get_imgs_from_data_cfg(data_cfg="./datanomot20.json")
    imgs_from_hie_datasets = get_imgs_from_hie_video_list(
        all_videos,
        hie_dataset_root="/fs/disk1/MOTDataset/HIE20")
    print(len(all_imgs))
    all_imgs.extend(imgs_from_nohie_datasets)
    print(len(all_imgs))
    # all_imgs.extend(imgs_from_hie_datasets)
    # print(len(all_imgs))

    imgs2coco(all_imgs, output_dataset_path="./pretrain_only_coco", ds_type="train")

    # # 2. 生成仅含hie的finetune coco格式数据集
    # imgs2coco(imgs_from_hie_datasets, output_dataset_path="./hie_finetune_coco", ds_type="train")