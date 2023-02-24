import csv
import pathlib
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import tod.utils.logger as logger
from tod.io import InputType
from tod.utils.misc import pair


def get_folders_from_fold_file(
    csv_file: str, path_prefix, test_fold: int, ignore: Optional[List[int]] = None
):
    path_prefix = pathlib.Path(path_prefix)
    test_datasets = []
    train_datasets = []
    if ignore is None:
        ignore = []

    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=",")

        # skip first comment line
        next(csv_reader)

        for row in csv_reader:
            if int(row[1]) in ignore:
                continue
            if int(row[1]) == test_fold:
                test_datasets.append(path_prefix / row[0])
            else:
                train_datasets.append(path_prefix / row[0])
    return train_datasets, test_datasets


class Latot(torch.utils.data.Dataset):

    def __init__(self, *, folders, config, transform: nn.Module = nn.Identity()):

        default_config = {
            "video_length": 50,  # chunk sizes of frames
            "crop_size": 512,  # crop in (h,w)
            "disjoint": False,
        }

        self.log = logger.getLogger('Latot')
        self.config = {**default_config, **config}

        crop = config["crop_size"]
        if type(crop) == int:
            self.config["crop_size"] = pair(crop)

        if type(folders) == str:
            folders = [folders]
        self.folders = folders

        # transform for images
        self.img_trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transform = transform

        self.num_channel = 3
        self.log.info("Channel size is {}".format(self.num_channel))
        self.__read_files(folders)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(folders={}, config={}, transform={})'.format(
                self.folders, self.config, self.transform
            )

    def image_size(self):
        return self.config["crop_size"]

    def num_channels(self):
        return self.num_channel

    def time_steps(self):
        return len(self.imgs[0])

    def __read_files(self, folders):
        self.imgs = []
        self.res = []
        self.gt = []
        for f in folders:
            self.log.debug("Processing folder {}".format(f))
            imgs = pathlib.Path(f) / "img"
            gt = pathlib.Path(f) / (pathlib.Path(f).stem + ".txt")

            self.__check_data(imgs, gt)

            ###########################################################
            # load file names and gt data
            ###########################################################
            # check consistency of sizes
            sizes = []

            gt = np.genfromtxt(gt, dtype=np.float32)
            gt = gt[:, :2] + gt[:, 2:] / 2
            self.log.debug("Found {} gt labels".format(len(gt)))
            sizes.append(len(gt))
            ids = self.__get_chunk_ids(gt)
            self.gt += self.__chunk(gt, ids)

            imgs = sorted(imgs.glob("*.jpg"))
            self.log.debug("Found {} image files".format(len(imgs)))
            sizes.append(len(imgs))
            chunks = self.__chunk(imgs, ids)
            self.imgs += chunks
            self.res += [Image.open(imgs[0]).size] * len(chunks)

            # TODO implement
            self.centers = None

            assert all(size == sizes[0] for size in sizes),\
                "Size mismatch between input and output"

        self.log.info("Finished processing. Collected {} elements".format(self.__len__()))

    # check if needed files/folders exists
    def __check_data(self, imgs, gt):

        assert imgs.exists(),\
            "imgs folder {} does not exist but is needed for input".format(imgs)

        assert gt.exists(), "gt {} does not exist".format(gt)

    # chunk data in exactly video_lengths sized chunks and filter gt
    def __get_chunk_ids(self, gt):
        n = self.config["video_length"]
        ids = np.arange(gt.shape[0])

        # no chunking needed, just return everything at once
        if n == 0:
            return np.empty(0)

        chunks = self.__get_chunks(gt)

        # return true, if a chunk from [a, b] should be filtered out
        def filter(a, b):
            # check if crop size is to small to capture current time steps
            bb = self.__bounding_box(gt[a:b, :2])
            if ((bb[1] - bb[0]) >= self.config["crop_size"]).any():
                return True
            # if this chunk contains ONLY invisibles, drop it
            if gt.shape[1] > 2 and gt[a:b, -1].all():
                return True
            # keep
            return False

        ids_list = []
        for (s, e) in chunks:
            if self.config["disjoint"]:
                ids_list += [
                    ids[i:i + n] for i in range(s, e - n, n) if not filter(i, i + n)
                ]
            else:
                ids_list += [
                    ids[i:i + n] for i in range(s, e - n) if not filter(i, i + n)
                ]
        return np.array(ids_list)

    def __get_chunks(self, gt):
        return [(0, len(gt))]

    # perform the chunking given the id list
    def __chunk(self, arr, filter_ids):
        if filter_ids.shape[0] == 0:
            if isinstance(arr, np.ndarray):
                return [a.reshape(1, -1) for a in arr]
            else:
                return [[a] for a in arr]

        if isinstance(arr, np.ndarray):
            return [arr[ids] for ids in filter_ids]

        return [[arr[id] for id in ids] for ids in filter_ids]

    # returns (min x, min y), (max x, max y)
    def __bounding_box(self, chunk):
        return (chunk.min(axis=0), chunk.max(axis=0))

    def __get_random_crop_tl(self, bb, img_size, eps=10):
        crop = self.config["crop_size"]

        a = max(bb[1][0] - crop[1], 0)
        b = min(bb[0][0], img_size[1] - crop[1])
        a = min(a + eps, b)
        b = max(b - eps, a)
        s = random.randint(a, b)

        a = max(bb[1][1] - crop[0], 0)
        b = min(bb[0][1], img_size[0] - crop[0])
        a = min(a + eps, b)
        b = max(b - eps, a)
        t = random.randint(a, b)

        return int(s), int(t)

    def __apply_sampler(self, idx):
        imgs = self.imgs[idx]
        res = self.res[idx]
        gt = self.gt[idx]
        return (imgs, res, gt)

    ####################################################
    #             accessors
    ####################################################
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        imgs_files, res, gt = self.__apply_sampler(idx)
        n = len(imgs_files)
        crop = self.config["crop_size"]

        gt = gt[:, :2]
        scale = min(res) / crop[0]
        real_target = np.ceil(np.array(res) / scale).astype(int)
        gt = (gt / scale).astype(int)
        bb = self.__bounding_box(gt)

        s, t = self.__get_random_crop_tl(bb, (real_target[1], real_target[0]))
        gt = (gt - (s, t)).astype(np.float32)
        if (gt < 0).any() or (gt > np.array(self.config["crop_size"])).any():
            self.log.warning("Warning: out of crop gt!")

        data = torch.empty(
            [n, self.num_channel, *self.config["crop_size"]], dtype=torch.float32
        )
        for i in range(n):
            self.log.debug("Reading image {}".format(imgs_files[i]))
            with Image.open(imgs_files[i]) as img:
                img = img.convert("RGB")
                img = img.resize(real_target)
                img = self.img_trans(img)
                data[i, :3] = img[:, t:t + crop[0], s:s + crop[1]]

        return self.transform([data, gt])
