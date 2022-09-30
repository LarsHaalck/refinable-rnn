import tod.utils.logger as logger
from tod.utils.misc import pair
from tod.io import InputType

import torch
import pathlib
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn as nn
from typing import List, Optional
import csv


def get_folders_from_fold_file(
    csv_file: str,
    path_prefix: str,
    test_fold: int,
    ignore: Optional[List[int]] = None
):
    path_prefix = pathlib.Path(path_prefix)
    test_datasets = []
    train_datasets = []
    if ignore is None:
        ignore = [0]

    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=",")

        # skip first comment line
        next(csv_reader)

        for row in csv_reader:
            if int(row[2]) in ignore:
                continue
            if int(row[2]) == test_fold:
                test_datasets.append(path_prefix / row[0])
            else:
                train_datasets.append(path_prefix / row[0])
    return train_datasets, test_datasets


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, *, folders, config, transform=nn.Identity(), sampler=None):

        default_config = {
            "input_type": InputType.ImagesUnaries,  # what input is used
            "video_length": 50,  # chunk sizes of frames
            "crop_size": 512,  # crop in (h,w)
            "resolution": (1080, 1920),  # in (h,w)
            "include_invisibles": False,
            "disjoint": False,
            "crop_center": False,
            "ignore_gt": False,
        }

        self.log = logger.getLogger('VideoDataset')
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

        # transform for unaries (need to be resized before
        self.unary_trans = transforms.Compose(
            [transforms.Resize(self.config["resolution"]), self.img_trans]
        )

        self.transform = transform

        self.num_channel = self.__has_img() * 3 + self.__has_unary()
        self.log.info("Channel size is {}".format(self.num_channel))
        self.__read_files(folders)

        self.sampler = sampler

        if self.config["ignore_gt"]:
            self.config["crop_center"] = True

    def __repr__(self):
        return self.__class__.__name__ + \
            '(folders={}, config={}, transform={}, sampler={})'.format(
                self.folders, self.config, self.transform, self.sampler
            )

    def image_size(self):
        return self.config["crop_size"]

    def num_channels(self):
        return self.num_channel

    def time_steps(self):
        if self.sampler is not None:
            return self.sampler.time_steps()

        if self.__has_img():
            return len(self.imgs[0])
        else:
            return len(self.unaries[0])

    def __read_files(self, folders):
        self.imgs = []
        self.unaries = []
        self.gt = []
        for f in folders:
            self.log.debug("Processing folder {}".format(f))
            imgs = pathlib.Path(f) / "imgs"
            unaries = pathlib.Path(f) / "unaries"
            centers = pathlib.Path(f) / "centers.csv"
            gt = pathlib.Path(f) / "gt_label.csv"

            self.__check_data(imgs, unaries, centers, gt)

            ###########################################################
            # load file names and gt data
            ###########################################################
            # check consistency of sizes
            sizes = []

            if self.config["ignore_gt"]:
                gt = np.empty(0)
                ids = np.empty(0)
            else:
                gt = np.genfromtxt(gt, delimiter=',', dtype=np.float32)
                self.log.debug("Found {} gt labels".format(len(gt)))
                sizes.append(len(gt))
                ids = self.__get_chunk_ids(gt)
                self.gt += self.__chunk(gt, ids)

            if self.__has_img():
                imgs = sorted(imgs.glob("*.png"))
                self.log.debug("Found {} image files".format(len(imgs)))
                sizes.append(len(imgs))
                self.imgs += self.__chunk(imgs, ids)

            if self.__has_unary():
                unaries = sorted(unaries.glob("*.png"))
                self.log.debug("Found {} unaries".format(len(unaries)))
                sizes.append(len(unaries))
                self.unaries += self.__chunk(unaries, ids)

            # TODO implement
            self.centers = None

            assert all(size == sizes[0] for size in sizes),\
                "Size mismatch between input and output"

        self.log.info("Finished processing. Collected {} elements".format(self.__len__()))

    # check if input type requires imgs folder
    def __has_img(self):
        return self.config["input_type"] in [InputType.Images, InputType.ImagesUnaries]

    # check if input type requires unary folder
    def __has_unary(self):
        return self.config["input_type"] in [InputType.Unaries, InputType.ImagesUnaries]

    # check if needed files/folders exists
    def __check_data(self, imgs, unaries, centers, gt):

        assert not self.__has_img() or imgs.exists(),\
            "imgs folder {} does not exist but is needed for input".format(imgs)

        assert not self.__has_unary() or unaries.exists(),\
            "unaries folder {} does not exist but is needed for input".format(unaries)

        if not self.config["ignore_gt"]:
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
        # skip until visible
        first = np.where(gt[:, -1] == 0)[0][0]
        gt_stripped = gt[first:]

        if self.config["include_invisibles"]:
            # find transitions
            ends = np.where(np.diff(gt_stripped[:, -1]) < 0)[0]
            if (gt_stripped.shape[0] - 1) not in ends:
                ends = np.append(ends, gt_stripped.shape[0] - 1)
            starts = np.sort(np.mod(ends + 1, gt_stripped.shape[0]))

            assert starts.shape == ends.shape, "Mismatch of starts and ends shape"
            chunks = [(starts[i] + first, ends[i] + first) for i in range(len(starts))]
        else:
            starts = np.where(np.diff(gt_stripped[:, -1]) < 0)[0] + 1
            starts = np.append(0, starts)

            ends = np.where(np.diff(gt_stripped[:, -1]) > 0)[0]
            if (gt_stripped.shape[0] - 1) not in ends:
                ends = np.append(ends, gt_stripped.shape[0] - 1)

            assert starts.shape == ends.shape, "Mismatch of starts and ends shape"
            chunks = [(starts[i] + first, ends[i] + first) for i in range(len(starts))]

        return chunks

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

    def __get_random_crop_tl(self, bb, eps=10):
        crop = self.config["crop_size"]
        img_size = self.config["resolution"]

        if self.config["crop_center"]:
            return (img_size[1] - crop[1]) // 2, (img_size[0] - crop[0]) // 2

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
        imgs = self.imgs[idx] if self.imgs else []
        unaries = self.unaries[idx] if self.unaries else []

        if self.config["ignore_gt"]:
            gt = []
        else:
            gt = self.gt[idx]

        if self.sampler is not None:
            return self.sampler([imgs, unaries, gt])

        return [imgs, unaries, gt]

    ####################################################
    #             accessors
    ####################################################
    def __len__(self):
        if self.__has_img():
            return len(self.imgs)
        return len(self.unaries)

    def __getitem__(self, idx):
        imgs_files, unaries_files, gt = self.__apply_sampler(idx)
        n = len(imgs_files) if self.imgs else len(unaries_files)

        if self.config["ignore_gt"]:
            label = np.empty(0)
            gt = np.empty(0)
            s, t = self.__get_random_crop_tl(None)
        else:
            label = gt[:, -1]
            gt = gt[:, :2]
            bb = self.__bounding_box(gt)
            s, t = self.__get_random_crop_tl(bb)
            gt = (gt - (s, t)).astype(np.float32)
            if (gt < 0).any() or (gt > np.array(self.config["crop_size"])).any():
                self.log.warning("fuck")

        crop = self.config["crop_size"]
        data = torch.empty(
            [n, self.num_channel, *self.config["crop_size"]], dtype=torch.float32
        )
        for i in range(n):
            if imgs_files:
                self.log.debug("Reading image {}".format(imgs_files[i]))
                with Image.open(imgs_files[i]) as img:
                    img = img.convert("RGB")
                    data[i, :3] = self.img_trans(img)[:, t:t + crop[0], s:s + crop[1]]
            if unaries_files:
                self.log.debug("Reading unary {}".format(unaries_files[i]))
                with Image.open(unaries_files[i]) as img:
                    img = img.convert("L")
                    data[i, -1] = self.unary_trans(img)[:, t:t + crop[0], s:s + crop[1]]

        return self.transform([data, label, gt])
