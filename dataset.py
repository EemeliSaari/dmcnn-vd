import os
import random
from zipfile import ZipFile

import cv2
import numpy as np
import requests
import torch
from colour_demosaicing import mosaicing_CFA_Bayer


BASE_URL = 'http://www.cmlab.csie.ntu.edu.tw/project/Deep-Demosaic/static/dataset.zip'


def download_data(path='data', url=BASE_URL):
    """

    """
    if not os.path.exists(path):
        os.mkdir(path)

    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        with requests.get(base_url, stream=True) as res:
            res.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in res.iter_content(chunk_size=2**14):
                    if chunk:
                        f.write(chunk)

    result_path = os.path.join(path, 'Flickr500')
    if not os.path.exists(result_path):
        with ZipFile(filename, 'r') as z:
            z.extractall(path)
    
    return os.path.abspath(result_path)


class ImagePatchDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, loader=None, sample_size=None, 
                 sample_idx=None, patch_size=(33, 33)):
        self.root = root
 
        self.transform = transform
        if not self.transform:
            self.transform = torch.from_numpy

        self.patch_size = patch_size

        self.loader = loader
        if not self.loader:
            self.loader = self._numpy_loader

        files = os.listdir(root)
        if sample_size:
            files = random.sample(files, sample_size)

        self.files_ = list(map(lambda x: os.path.join(root, x), files))
        self.images_ = list(map(lambda x: np.array(self.loader(x)), self.files_))
        self.cfa_ = list(map(self._mosaic, self.images_))
        self.patches_ = self._compute_patches(self.images_)

    def __getitem__(self, idx):
        patch, img_id = self.patches_[idx]

        x, y = patch
        b0, b1 = self.patch_size
        truth = self.images_[img_id][x - b0:x, y - b1:y, :]
        cfa = self.cfa_[img_id][x - b0:x, y - b1:y].reshape((3, 33, 33)) / 255

        truth = cv2.resize(truth, (21, 21)).reshape((3, 21, 21)) / 255

        if self.transform:
            truth = self.transform(truth)
            cfa = self.transform(cfa)

        return cfa, truth

    def __len__(self):
        return len(self.patches_)

    def _compute_patches(self, images):

        patches = []
        for idx, img in enumerate(images):

            M, N, Z = img.shape
            b0, b1 = self.patch_size

            for i in range(b0, M-b0, 5):
                for j in range(b1, N-b1, 5):
                    patches.append(([i, j], idx))
        return patches

    def _numpy_loader(self, path):
        return cv2.imread(path, 1)

    def _mosaic(self, img):
        cfa = np.zeros(img.shape, np.uint8)
        mosaic = mosaicing_CFA_Bayer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for i in range(1):#cfa.shape[2]):
            cfa[:, :, i] = mosaic
        return cfa

