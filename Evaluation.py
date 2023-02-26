import numpy as np
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
from scipy.ndimage import zoom
import math
import cv2

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)*2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_computation(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_computation(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_computation(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    path_infer = "/media/NAS02/lc/braTS_transfer/braTS_s_cycle_npy/"
    # path_infer = "/media/NAS02/lc/braTS_transfer/braTS_s_pix2pix_npy/"
    # path_infer = "/media/NAS02/lc/braTS_transfer/MUNIT_00/"
    # path_infer = "/media/NAS02/lc/braTS_transfer/MUNIT_03/"
    # path_infer = "/media/NAS02/lc/BraTS_transfer_1/BraTS_s_pix2pix_npy/"
<<<<<<< HEAD
    # path_infer = "/media/NAS02/lc/BraTS_transfer_1/braTS_s_cycle_npy/"
    # path_infer = "/media/NAS02/lc/BraTS_transfer_1/MUNIT_00/"
    # r"C:\Users\wlc\PycharmProjects\pytorch-CycleGAN-and-pix2pix\results_npy"
    path_infer = "/media/NAS02/lc/braTS_transfer/braTS_s_pix2pix_norm_npy/"
    path_gt = "/media/NAS02/BraTS2020/pix2pix_cycle/style_transfer_divided/test/"
=======
    path_infer = "/media/NAS02/lc/BraTS_transfer_1/braTS_s_cycle_npy/"
    # path_infer = "/media/NAS02/lc/BraTS_transfer_1/MUNIT_00/"
    # r"C:\Users\wlc\PycharmProjects\pytorch-CycleGAN-and-pix2pix\results_npy"
    # path_infer = "/media/NAS02/lc/braTS_transfer/braTS_s_pix2pix_norm_npy/"
    # path_infer = "/home/lichao/MSPC/results/media/ssd/lc/BraTS2020/pix2pix_cycle/style_transfer_divided_unpaire/maxgcpert3_gan/12_256_AtoB_resnet_9blocks_basic_2_0.1_1.0_unbounded_0.5/test_latest/images/fake_Bnpy"
    # path_gt = "/media/NAS02/BraTS2020/pix2pix_cycle/style_transfer_divided/test/"
    path_gt = "/media/NAS02/BraTS2020/pix2pix_cycle/style_transfer_divided_unpaired/testB/"
    # /media/NAS02/BraTS2020/pix2pix_cycle/style_transfer_divided_unpaired/testB/
>>>>>>> b80cd9c50abe58b2a21f39d46481d189ca292ab3
    # r"D:\style_transfer_divided_unpaired\testB"

    path_gts = []
    path_infers = []
    for root, _, fnames in sorted(os.walk(path_gt)):
        for fname in fnames:
            path = os.path.join(root, fname)
            path_gts.append(path)
<<<<<<< HEAD
    # print(path_gts)
=======
    print(path_gts)
>>>>>>> b80cd9c50abe58b2a21f39d46481d189ca292ab3
    # print(len(path_gts))

    for root, _, fnames in sorted(os.walk(path_infer)):
        for fname in fnames:
            path = os.path.join(root, fname)
            path_infers.append(path)
    # print(path_infers)
    # print(len(path_infers))

    psnrs = []
    ssims = []
    fids = []
<<<<<<< HEAD
=======
    path_infers = sorted(path_infers)
    path_gts = sorted(path_gts)
>>>>>>> b80cd9c50abe58b2a21f39d46481d189ca292ab3
    for idx, (infer_img_path, gt_img_path) in enumerate(zip(path_infers, path_gts)):
        if idx == 500:
            break
        print(f"{idx}: {infer_img_path}, {gt_img_path}.")
        infer_img = np.load(infer_img_path, allow_pickle=True)
<<<<<<< HEAD
        # print(infer_img.shape)
        if len(infer_img.shape) == 4:
            infer_img = infer_img[0][0]
        gt_img = np.load(gt_img_path)
        if gt_img.max() > 0:
=======
        #print(infer_img.shape)
        if len(infer_img.shape) == 3:
            infer_img = infer_img[0]
        if len(infer_img.shape) == 4:
            infer_img = infer_img[0][0]

        gt_img = np.load(gt_img_path)
        
        if gt_img.max() > 0:
            # print(gt_img.shape)
>>>>>>> b80cd9c50abe58b2a21f39d46481d189ca292ab3
            if len(gt_img.shape) == 3:
                gt_img = gt_img[0]
            if gt_img.shape[0] == 480:
                gt_img = gt_img[240:480, :]
            
            zoom_scale_H, zoom_scale_W = infer_img.shape[0]/gt_img.shape[0], infer_img.shape[1]/gt_img.shape[1]
<<<<<<< HEAD
            gt_img = zoom(gt_img, zoom=[zoom_scale_H, zoom_scale_W], order=3)
            #print(gt_img.shape)
            fid = 0
            # fid = calculate_fid((infer_img - infer_img.min())/(infer_img.max() - infer_img.min()), (gt_img - gt_img.min())/(gt_img.max() - gt_img.min()))
            gt_img = ((gt_img - gt_img.min())/(gt_img.max() - gt_img.min())) * 255
            infer_img = ((infer_img - infer_img.min())/(infer_img.max() - infer_img.min())) * 255
            # print(infer_img.max())
            # print(gt_img.max())
=======
            infer_img = zoom(infer_img, zoom=[1/zoom_scale_H, 1/zoom_scale_W], order=3)
            # print(gt_img.shape)
            fid = 0
            # fid = calculate_fid((infer_img - infer_img.min())/(infer_img.max() - infer_img.min()), (gt_img - gt_img.min())/(gt_img.max() - gt_img.min()))
            # gt_img = ((gt_img - gt_img.min())/(gt_img.max() - gt_img.min())) * 255
            #infer_img = ((infer_img - infer_img.min())/(infer_img.max() - infer_img.min())) * 255
            infer_img = infer_img * 255

            print(infer_img.max())
            print(gt_img.max())
>>>>>>> b80cd9c50abe58b2a21f39d46481d189ca292ab3
            psnr, ssim = calculate_psnr(infer_img, gt_img), calculate_ssim(infer_img, gt_img)
            
            print(f"PSNR: {psnr}, SSIM: {ssim}, FID: {fid}")
            psnrs.append(psnr)
            ssims.append(ssim)
            fids.append(fid)
    psnrs, ssims, fids = np.array(psnrs), np.array(ssims), np.array(fids)
    print("number of tested cases: ", len(psnrs))
    print("psnr: ", psnrs.mean())
    print("ssim: ", ssims.mean())
    print("fid: ", fids.mean())






# def read_img(mod_1_path, mod_2_path, src_file_name, mod_1_save_path, mod_2_save_path, mod_12_save_path, gt_file_path, gt_file_name, gt_save_path):  # for .mhd .nii .nrrd
#     '''
#     N*h*W
#     :param full_path_filename:
#     :return:*H*W
#     '''
#     if not os.path.exists(mod_1_path):
#         raise FileNotFoundError
#     mod_1_data = nib.load(mod_1_path).get_fdata()
#     mod_2_data = nib.load(mod_2_path).get_fdata()
#
#     mod_1_data_norm = (mod_1_data - mod_1_data.mean()) / (mod_1_data.std())  # case norm
#     mod_2_data_norm = (mod_2_data - mod_2_data.mean()) / (mod_2_data.std())
#
#     mod_1_data_norm = ((mod_1_data - mod_1_data.min()) / (mod_1_data.max() - mod_1_data.min())) * 255  # case norm
#     mod_2_data_norm = ((mod_2_data - mod_2_data.min()) / (mod_2_data.max() - mod_2_data.min())) * 255
#
#     print(src_file_name, " data_shape:", mod_1_data_norm.shape)
#     print(mod_1_data_norm.max())
#     if not os.path.exists(gt_file_path):
#         raise FileNotFoundError
#
#     mask_data = nib.load(gt_file_path).get_fdata()
#     mask_data_norm = mask_data  # no case norm for gt_seg
#
#     save_mod_1_path_npy = mkdir(os.path.join(mod_1_save_path, src_file_name))
#     save_mod_2_path_npy = mkdir(os.path.join(mod_2_save_path, src_file_name))
#     save_mod_12_path_npy = mkdir(os.path.join(mod_12_save_path, src_file_name))
#     save_gt_path_npy = mkdir(os.path.join(gt_save_path, gt_file_name))
#
#     num_slices = 0
#     for slice_idx in range(0, mask_data_norm.shape[2]):
#         mod_1_slice = mod_1_data_norm[:, :, slice_idx].reshape((1, 240, 240))
#         mod_2_slice = mod_2_data_norm[:, :, slice_idx].reshape((1, 240, 240))
#         combine_slice = np.concatenate((mod_1_slice, mod_2_slice), axis=1)
#         mask_slice = mask_data_norm[:, :, slice_idx]
#         mask_slice = grayval2label(mask_slice)
#
#         num_slices += 1
#         np.save(os.path.join(save_mod_1_path_npy, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), mod_1_slice)
#         np.save(os.path.join(save_mod_2_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mod_2_slice)
#         np.save(os.path.join(save_mod_12_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), combine_slice)
#         np.save(os.path.join(save_gt_path_npy, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), mask_slice)
#
#         num_slices += 1
#     print(num_slices)
#
#
# # def read_img(mod_1_path, mod_2_path, src_file_name, mod_1_save_path, mod_2_save_path, mod_12_save_path, gt_file_path, gt_file_name, gt_save_path):  # for .mhd .nii .nrrd
#
# def read_dataset(mod_1_paths, mod_2_paths, gt_file_paths, mod_1_save_path, mod_2_save_path, mod_12_save_path, gt_save_path):
#     for idx_data in range(len(mod_1_paths)):
#         print('{} / {}'.format(idx_data + 1, len(mod_1_paths)))
#         mod_1_path = mod_1_paths[idx_data]
#         mod_2_path = mod_2_paths[idx_data]
#         mask_path = gt_file_paths[idx_data]
#
#         nameext, _ = os.path.splitext(mod_1_path)
#         nameext, _ = os.path.splitext(nameext)
#
#         mask_nameext, _ = os.path.splitext(mask_path)
#         mask_nameext, _ = os.path.splitext(mask_nameext)
#
#         _, name = os.path.split(nameext)
#         src_name = name[:-6]
#         _, mask_name = os.path.split(mask_nameext)
#         print(mask_name)
#         print(src_name)
#
#         read_img(mod_1_path, mod_2_path, src_name, mod_1_save_path, mod_2_save_path, mod_12_save_path, mask_path, mask_name, gt_save_path)
#
#
# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     return path
#
#
# if __name__ == '__main__':
#
#     # path_raw_dataset_type = r"D:\braTS_t"
#     # mod_1_path_save = mkdir(r"D:\style_transfer_test\A")
#     # mod_12_path_save = mkdir(r"D:\style_transfer_test\AB")
#     # mod_2_path_save = mkdir(r"D:\style_transfer_test\B")
#     # gt_path_save = mkdir(r"D:\style_transfer_test\gt")
#
#     path_raw_dataset_type = "/media/NAS02/BraTS2020/Training/"
#     mod_1_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/A")
#     mod_12_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/AB")
#     mod_2_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/B")
#     gt_path_save = mkdir("/media/NAS02/BraTS2020/pix2pix_cycle/gt")
#
#     mask_paths = []
#     t2_paths = []
#     t1_paths = []
#     t1ce_paths = []
#     flair_paths = []
#     start = 151
#     start = 0
#     for p_id in range(start, 1667):
#         p_id_str = str(p_id)
#         path_raw_dataset_type_patient = os.path.join(path_raw_dataset_type, 'BraTS20_Training_'+p_id_str.zfill(3))
#         # print(path_raw_dataset_type_patient)
#         flair_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*flair.nii')))
#         t1_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t1.nii')))
#         t1ce_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t1ce.nii')))
#         t2_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*t2.nii')))
#         mask_paths.extend(glob(os.path.join(path_raw_dataset_type_patient, '*seg.nii')))
#
#     print(flair_paths)
#     print(len(flair_paths))
#
#     read_dataset(flair_paths, t1_paths, mask_paths, mod_1_path_save, mod_2_path_save, mod_12_path_save, gt_path_save)
#
