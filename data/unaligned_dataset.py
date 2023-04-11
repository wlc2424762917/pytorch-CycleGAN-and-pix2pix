import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from scipy.ndimage import zoom
import torch


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        A = np.load(A_path)
        B = np.load(B_path)
        # A, B = A/255., B/255.
        A = A/255
        B = B/255
        # transform.norm(0.5, 0.5)
        # A = (A - 0.5) / 0.5
        # B = (B - 0.5) / 0.5
        w, h = A.shape

        transform_params = get_params(self.opt, A.shape)
        crop_pos, flip = transform_params['crop_pos'], transform_params['flip']

        #### rescale ####
        zoom_scale_H, zoom_scale_W = self.opt.load_size / h, self.opt.load_size / w
        A = zoom(A, zoom=[zoom_scale_H, zoom_scale_W], order=3)
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

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}