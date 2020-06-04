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
            labels = np.loadtxt(l, dtype=np.float32).reshape(-1, 6)[:, 2:6]
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



if __name__ == "__main__":
    train_videos=["2", "4", "6", "7", "8", "10", "11", "13", "14", "15", "16", "18"]
    val_videos=["1", "3", "5", "9", "12", "17", "19"]
    all_videos=[
        "/fs/disk1/MOTDataset/HIE20/images/train/2/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/4/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/6/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/7/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/8/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/10/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/11/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/13/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/14/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/15/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/16/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/18/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/1/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/3/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/5/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/9/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/12/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/17/img",
        "/fs/disk1/MOTDataset/HIE20/images/train/19/img",
    ]

    hie2coco(videos=train_videos,
             output_dir="./hie_coco_part",
             root="/fs/disk1/MOTDataset/HIE20",
             src_img_dir="images/train",
             ds_type="train")
    hie2coco(videos=val_videos,
             output_dir="./hie_coco_part",
             root="/fs/disk1/MOTDataset/HIE20",
             src_img_dir="images/train",
             ds_type="val")

    hie2coco(videos=all_videos,
             output_dir="./hie_coco_all",
             root="/fs/disk1/MOTDataset/HIE20",
             src_img_dir="images/train",
             ds_type="train")
    hie2coco(videos=val_videos,
             output_dir="./hie_coco_all",
             root="/fs/disk1/MOTDataset/HIE20",
             src_img_dir="images/train",
             ds_type="val")