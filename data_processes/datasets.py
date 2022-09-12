import os
import h5py
import json

import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """
    Dataset class to be used by the DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transformation pipeline
        """
        self.split = split
        assert split in {"TRAIN", "VAL", "TEST"}

        # Load hdf5 file where images are stored
        self.h = h5py.File(
            os.path.join(data_folder, split + "_IMAGES_" + data_name + ".hdf5"),
            "r",
        )
        self.imgs = self.h["images"]
        self.cpi = self.h.attrs["captions_per_image"]

        # Load encoded captions and caption lengths
        with open(
            os.path.join(data_folder, split + "_CAPTIONS_" + data_name + ".json"),
            "r",
        ) as j:
            self.captions = json.load(j)

        with open(
            os.path.join(data_folder, split + "_CAPLENS_" + data_name + ".json"),
            "r",
        ) as j:
            self.caplens = json.load(j)

        self.transform = transform
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.0)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == "TRAIN":
            return img, caption, caplen

        all_captions = torch.LongTensor(
            self.captions[
                ((i // self.cpi) * self.cpi) : (((i // self.cpi) * self.cpi) + self.cpi)
            ]
        )
        return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
