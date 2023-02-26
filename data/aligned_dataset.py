import os
import random

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory

        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]

        # AB = Image.open(AB_path).convert('RGB')
        AB = np.load(AB_path)
        AB = AB[0]

        # split AB image into A and B
        w, h = AB.shape
        w2 = int(w / 2)

        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))
        A = AB[0:w2, 0:h]
        B = AB[w2:w, 0:h]

<<<<<<< HEAD
        A = (A - A.min()) / (A.max() - A.min() + 1e-6)
        B = (B - B.min()) / (B.max() - B.min() + 1e-6)

        # transform.norm(0.5, 0.5)
        # A = (A - 0.5) / 0.5
        # B = (B - 0.5) / 0.5
=======
        # A = (A - A.min()) / (A.max() - A.min() + 1e-6)
        # B = (B - B.min()) / (B.max() - B.min() + 1e-6)
        A, B  = A/255, B/255

        # transform.norm(0.5, 0.5)
        A = (A - 0.5) / 0.5
        B = (B - 0.5) / 0.5
>>>>>>> b80cd9c50abe58b2a21f39d46481d189ca292ab3

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.shape)
        crop_pos, flip = transform_params['crop_pos'], transform_params['flip']
        # print(transform_params)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        # A = A_transform(A)
        # B = B_transform(B)
        #### rescale ####
        zoom_scale_H, zoom_scale_W = self.opt.load_size/h, self.opt.load_size/w2
        A = zoom(A,  zoom=[zoom_scale_H, zoom_scale_W], order=3)
        B = zoom(B, zoom=[zoom_scale_H, zoom_scale_W], order=3)
        #### crop ####
        crop_size = self.opt.crop_size
        A = A[crop_pos[0]: crop_pos[0] + crop_size, crop_pos[1]: crop_pos[1] + crop_size]
        B = B[crop_pos[0]: crop_pos[0] + crop_size, crop_pos[1]: crop_pos[1] + crop_size]
        #### flip ####
        if flip:
            hori = random.random()
            if hori > 0.5:
                A = np.flip(A, 1)
                B = np.flip(B, 1)
            else:
                A = np.flip(A, 0)
                B = np.flip(B, 0)
        A, B = np.expand_dims(A, 0), np.expand_dims(B, 0)
        A = torch.from_numpy(np.ascontiguousarray(A)).float()
        B = torch.from_numpy(np.ascontiguousarray(B)).float()
        # print(A.shape)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


if __name__ == "__main__":
    dir_AB = os.path.join(r"D:\style_transfer_test\AB")  # get the image directory

    # AB_paths = sorted(make_dataset(dir_AB, 10))  # get image paths
    # print(AB_paths)

