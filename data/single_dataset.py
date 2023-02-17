from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from scipy.ndimage import zoom
import torch


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        # A = self.transform(A_img)
        A = np.load(A_path)
        A= A / 255.
        w, h = A.shape

        transform_params = get_params(self.opt, A.shape)
        crop_pos, flip = transform_params['crop_pos'], transform_params['flip']

        #### rescale ####
        zoom_scale_H, zoom_scale_W = self.opt.load_size / h, self.opt.load_size / w
        A = zoom(A, zoom=[zoom_scale_H, zoom_scale_W], order=3)
        #### crop ####
        crop_size = self.opt.crop_size
        A = A[crop_pos[0]: crop_pos[0] + crop_size, crop_pos[1]: crop_pos[1] + crop_size]
        #### flip ####
        if flip:
            hori = random.random()
            if hori > 0.5:
                A = np.flip(A, 1)
            else:
                A = np.flip(A, 0)
        A = np.expand_dims(A, 0)
        A = torch.from_numpy(np.ascontiguousarray(A)).float()

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


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

    # flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}