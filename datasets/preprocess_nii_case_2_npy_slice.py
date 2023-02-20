import os
import numpy as np
from glob import glob
from skimage.transform import resize
import SimpleITK as sitk
from matplotlib import pylab as plt
import nibabel as nib
from collections import OrderedDict
from skimage.morphology import label
import pickle


def grayval2label(gray_mask):
    gray_mask = np.squeeze(gray_mask)
    gray_val_list = [0, 1, 2, 4]
    map_gray2label = {}
    for i, key in enumerate(gray_val_list):
        map_gray2label[key] = i
    label_mask = np.zeros_like(gray_mask)
    for row in range(gray_mask.shape[0]):
        for col in range(gray_mask.shape[1]):
            label_mask[row][col] = map_gray2label[gray_mask[row][col]]
    return label_mask.reshape((1, gray_mask.shape[0], gray_mask.shape[1]))


def get_image_properties(src_img, gt_seg, all_classes, class_connection):
    # we need to find out where the classes are and sample some random locations
    # let's do 10.000 samples per class
    # seed this for reproducibility!
    properties = OrderedDict()
    num_samples = 10000
    min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
    rndst = np.random.RandomState(1234)
    class_locs = {}
    for c in all_classes:
        all_locs = np.argwhere(gt_seg == c)
        if len(all_locs) == 0:
            class_locs[c] = []
            continue
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))
        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[c] = selected
    properties['class_locations'] = class_locs

    properties['class_connection'] = class_connection
    return properties


def read_img(mod_1_path, mod_2_path, src_file_name, mod_1_save_path, mod_2_save_path, mod_12_save_path, gt_file_path, gt_file_name, gt_save_path):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    if not os.path.exists(mod_1_path):
        raise FileNotFoundError
    mod_1_data = nib.load(mod_1_path).get_fdata()
    mod_2_data = nib.load(mod_2_path).get_fdata()

    mod_1_data_norm = (mod_1_data - mod_1_data.mean()) / (mod_1_data.std())  # case norm
    mod_2_data_norm = (mod_2_data - mod_2_data.mean()) / (mod_2_data.std())

    mod_1_data_norm = ((mod_1_data - mod_1_data.min()) / (mod_1_data.max() - mod_1_data.min())) * 255  # case norm
    mod_2_data_norm = ((mod_2_data - mod_2_data.min()) / (mod_2_data.max() - mod_2_data.min())) * 255

    print(src_file_name, " data_shape:", mod_1_data_norm.shape)
    print(mod_1_data_norm.max())
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError

    mask_data = nib.load(gt_file_path).get_fdata()
    mask_data_norm = mask_data  # no case norm for gt_seg

    save_mod_1_path_npy = mkdir(os.path.join(mod_1_save_path, src_file_name))
    save_mod_2_path_npy = mkdir(os.path.join(mod_2_save_path, src_file_name))
    save_mod_12_path_npy = mkdir(os.path.join(mod_12_save_path, src_file_name))
    save_gt_path_npy = mkdir(os.path.join(gt_save_path, gt_file_name))

    num_slices = 0
    for slice_idx in range(0, mask_data_norm.shape[2]):
        mod_1_slice = mod_1_data_norm[:, :, slice_idx].reshape((1, 240, 240))
        mod_2_slice = mod_2_data_norm[:, :, slice_idx].reshape((1, 240, 240))
        combine_slice = np.concatenate((mod_1_slice, mod_2_slice), axis=1)
        mask_slice = mask_data_norm[:, :, slice_idx]
        mask_slice = grayval2label(mask_slice)

        num_slices += 1
        np.save(os.path.join(save_mod_1_path_npy, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), mod_1_slice)
        np.save(os.path.join(save_mod_2_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mod_2_slice)
        np.save(os.path.join(save_mod_12_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), combine_slice)
        np.save(os.path.join(save_gt_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mask_slice)

        num_slices += 1
    print(num_slices)


# def read_img(mod_1_path, mod_2_path, src_file_name, mod_1_save_path, mod_2_save_path, mod_12_save_path, gt_file_path, gt_file_name, gt_save_path):  # for .mhd .nii .nrrd

def read_dataset(mod_1_paths, mod_2_paths, gt_file_paths, mod_1_save_path, mod_2_save_path, mod_12_save_path, gt_save_path):
    for idx_data in range(len(mod_1_paths)):
        print('{} / {}'.format(idx_data + 1, len(mod_1_paths)))
        mod_1_path = mod_1_paths[idx_data]
        mod_2_path = mod_2_paths[idx_data]
        mask_path = gt_file_paths[idx_data]

        nameext, _ = os.path.splitext(mod_1_path)
        nameext, _ = os.path.splitext(nameext)

        mask_nameext, _ = os.path.splitext(mask_path)
        mask_nameext, _ = os.path.splitext(mask_nameext)

        _, name = os.path.split(nameext)
        src_name = name[:-6]
        _, mask_name = os.path.split(mask_nameext)
        print(mask_name)
        print(src_name)

        read_img(mod_1_path, mod_2_path, src_name, mod_1_save_path, mod_2_save_path, mod_12_save_path, mask_path, mask_name, gt_save_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':

    # path_raw_dataset_type = r"D:\braTS_t"
    # mod_1_path_save = mkdir(r"D:\style_transfer_test\A")
    # mod_12_path_save = mkdir(r"D:\style_transfer_test\AB")
    # mod_2_path_save = mkdir(r"D:\style_transfer_test\B")
    # gt_path_save = mkdir(r"D:\style_transfer_test\gt")

    path_raw_dataset_type = "/media/NAS02/BraTS2020/Training/"
    mod_1_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/A")
    mod_12_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/AB")
    mod_2_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/B")
    gt_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/gt")

    mask_paths = []
    t2_paths = []
    t1_paths = []
    t1ce_paths = []
    flair_paths = []
    start = 151
    start = 0
    for p_id in range(start, 1667):
        p_id_str = str(p_id)
        path_raw_dataset_type_patient = os.path.join(path_raw_dataset_type, 'BraTS20_Training_'+p_id_str.zfill(3))
        # print(path_raw_dataset_type_patient)
        flair_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*flair.nii')))
        t1_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t1.nii')))
        t1ce_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t1ce.nii')))
        t2_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t2.nii')))
        mask_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*seg.nii')))

    print(flair_paths)
    print(len(flair_paths))

    read_dataset(flair_paths, t1_paths, mask_paths, mod_1_path_save, mod_2_path_save, mod_12_path_save, gt_path_save)

