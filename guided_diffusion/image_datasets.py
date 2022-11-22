import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    scale_init=1.0,
    scale_factor=0.75,
    stop_scale=16,
    current_scale=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = ImageDataset(
        image_size,
        data_dir,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        scale_init=scale_init,
        scale_factor=scale_factor,
        stop_scale=stop_scale,
        current_scale=current_scale,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_path,
        classes=None,
        shard=0,
        num_shards=1,
        scale_init=1.0,
        scale_factor=0.75,
        stop_scale=16,
        current_scale=0,
        blur_lr_image=True
    ):
        super().__init__()
        self.resolution = resolution
        with bf.BlobFile(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        current_factor = scale_init * math.pow(scale_factor, stop_scale-current_scale)
        curr_w, curr_h = round(pil_image.size[0] * current_factor), round(pil_image.size[1] * current_factor)
        self.pil_image = pil_image.resize((curr_w, curr_h))
        self.pil_image = self.pil_image.convert("RGB")
        if current_scale != 0:
            lr_scale = scale_init * math.pow(scale_factor, stop_scale-current_scale + 1)
            pil_image_lr = pil_image.resize((round(pil_image.size[0] * lr_scale), round(pil_image.size[1] * lr_scale)))
            pil_image_lr = pil_image_lr.resize((round(pil_image.size[0] * current_factor), round(pil_image.size[1] * current_factor)))
            self.pil_image_lr = pil_image_lr.convert("RGB")
        else:
            self.pil_image_lr = None
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.blur_lr_image = blur_lr_image

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        arr = np.array(self.pil_image, dtype=np.float32)
        arr_lr = np.array(self.pil_image_lr, dtype=np.float32)

        if self.blur_lr_image:
            arr_lr = cv2.GaussianBlur(arr_lr, ksize=(3, 3), sigmaX=0.6, sigmaY=0.6)

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        # if self.pil_image_lr is not None:
        #     out_dict["y"] = np.transpose(arr_lr.astype(np.float32) / 127.5 - 1, [2, 0, 1])

        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size, crop_size):
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - crop_size) // 2
    crop_x = (arr.shape[1] - crop_size) // 2
    arr = arr[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
    arr = cv2.resize(arr, (image_size, image_size))
    return arr, (crop_y, crop_x)


def random_crop_arr(pil_image, image_size, crop_size, xy=None):
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - crop_size + 1) if xy is None else xy[0]
    crop_x = random.randrange(arr.shape[1] - crop_size + 1) if xy is None else xy[1]
    arr = arr[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
    arr = cv2.resize(arr, (image_size, image_size))
    return arr, (crop_y, crop_x)
