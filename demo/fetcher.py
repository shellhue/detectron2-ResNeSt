import os

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