import numpy as np
import sys
sys.path.append('../')
from scipy.io import savemat
import os
import matplotlib.pyplot as plt
import scipy
from skimage.measure import compare_ssim


def removeFEOversampling(src):
    """ Remove Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    assert src.ndim >= 2
    nFE, nPE = src.shape[-2:]
    if nPE != nFE:
        return np.take(src, np.arange(int(nFE*0.25)+1, int(nFE*0.75)+1), axis=-2)
    else:
        return src

def addFEOversampling(src):
    """ Add Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    shape = list(src.shape)
    shape_upper = shape.copy()
    shape_upper[-2] = shape[-2] // 2 + 1
    shape_lower = shape.copy()
    shape_lower[-2] = shape[-2] // 2 - 1
    zeros_upper = np.zeros(tuple(shape_upper), src.dtype)
    zeros_lower = np.zeros(tuple(shape_lower), src.dtype)
    dst = np.concatenate((zeros_upper, src, zeros_lower), axis=-2)
    return dst

def removePEOversampling(src):
    """ Remove Phase Encoding (PE) oversampling. """
    nPE = src.shape[-1]
    nFE = src.shape[-2]
    PE_OS_crop = (nPE - nFE) / 2

    if PE_OS_crop == 0:
        return src
    else:
        return np.take(src, np.arange(int(PE_OS_crop)+1, nPE-int(PE_OS_crop)+1), axis=-1)

def fft2c(img):
    """ Centered fft2 """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))) / np.sqrt(img.shape[-2]*img.shape[-1])

def ifft2c(img):
    """ Centered ifft2 """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img))) * np.sqrt(img.shape[-2]*img.shape[-1])

def mriAdjointOp(rawdata, coilsens, mask):
    """ Adjoint MRI Cartesian Operator """
    return np.sum(ifft2c(rawdata * mask)*np.conj(coilsens), axis=0)

def mriForwardOp(img, coilsens, mask):
    """ Forward MRI Cartesian Operator """
    return fft2c(coilsens * img)*mask

def saveAsMat(img, filename, matlab_id, mat_dict=None):
    """ Save mat files with ndim in [2,3,4]

        Args:
            img: image to be saved
            file_path: base directory
            matlab_id: identifer of variable
            mat_dict: additional variables to be saved
    """
    assert img.ndim in [2, 3, 4]

    img_normalized = img.copy()
    if img.ndim == 3:
        img_normalized = np.transpose(img_normalized, (1, 2, 0))
    elif img.ndim == 4:
        img_normalized = np.transpose(img_normalized, (2, 3, 0, 1))

    if mat_dict == None:
        mat_dict = {matlab_id: img_normalized}
    else:
        mat_dict[matlab_id] = img_normalized

    dirname = os.path.dirname(filename) or '.'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savemat(filename, mat_dict)

def _normalize(img):
    """ Normalize image between [0, 1] """
    tmp = img - np.min(img)
    tmp /= np.max(tmp)
    return tmp

def kshow(kspace):
    """ Visualize kspace (logarithm). """
    img = np.abs(kspace)
    img /= np.max(img)
    img = np.log(img + 1e-5)
    plt.figure();
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')


def ksave(kspace, filepath):
    """ Save kspace (logarithm). """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    img = np.abs(kspace)
    img /= np.max(img)
    img = np.log(img + 1e-5)
    scipy.misc.imsave(filepath, _normalize(img).astype(np.uint8))

def imshow(img, title=""):
    """ Show image as grayscale. """
    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)

def phaseshow(img, title=''):
    """ Show phase of image. """
    if not (img.dtype == np.complex64 or img.dtype == np.complex128):
        print('img is not complex!')
    img = np.angle(img)

    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    plt.set_cmap('hsv')

def postprocess(img, dataset):
    """ Postprocess NYU Knee data.
    For other postprocessing, please add your postprocessing steps here."""
    if dataset in ['coronal_pd', 'axial_t2', 'coronal_pd_fs', 'sagittal_pd', 'sagittal_t2']:
        img = removePEOversampling(img)
    else:
        print(Warning("Postprocessing not defined for dataset %s" % dataset))

    assert img.ndim in [2, 3]
    img_ndim = img.ndim
    if img_ndim == 2:
        img = img[np.newaxis]

    for i in range(img.shape[0]):
        if dataset in ['coronal_pd', 'axial_t2', 'coronal_pd_fs']:
            img[i] = np.flipud(np.fliplr(img[i]))
        elif dataset in ['sagittal_pd', 'sagittal_t2']:
            img[i] = np.flipud(np.rot90(img[i]))
        else:
            print(Warning("Postprocessing not defined for dataset %s" % dataset))
    if img_ndim == 2:
        img = img[0]

    return img

def contrastStretching(img, saturated_pixel=0.004):
    """ constrast stretching according to imageJ
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    v_min = values[lim]
    v_max = values[-lim-1]
    img = (img - v_min)*(255.0)/(v_max - v_min)
    img = np.minimum(255.0, np.maximum(0.0, img))
    return img

def brighten(img, beta):
    """ brighten image according to Matlab."""
    if np.max(img) > 1:
        img / 255.0

    assert beta > 0 and beta < 1
    tol = np.sqrt(2.2204e-16)
    gamma = 1 - min(1-tol, beta)
    img = img ** gamma
    return img

def getContrastStretchingLimits(img, saturated_pixel=0.004):
    """ constrast stretching according to imageJ
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm"""
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    v_min = values[lim]
    v_max = values[-lim-1]
    return v_min, v_max

def normalize(img, v_min, v_max, max_int=255.0):
    """ normalize image to [0, max_int] according to image intensities [v_min, v_max] """
    img = (img - v_min)*(max_int)/(v_max - v_min)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img

def imsave(img, filepath, normalize=True):
    """ Save an image. """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    if normalize:
        img = _normalize(img)
        img *= 255.0
    scipy.misc.imsave(filepath, img.astype(np.uint8))

def imsaveDiff(img, maxIntensity, scale, filepath):
    """ Save difference image according to maxIntensity. Amplify difference by scale. """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    if img.dtype == np.complex:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    tmp = img
    tmp /= maxIntensity
    tmp *= scale
    tmp = np.minimum(tmp, 1) * 255.0

    scipy.misc.imsave(filepath, tmp.astype(np.uint8))

def rmse(img, ref):
    """ Compute RMSE. If inputs are 3D, average over axis=0 """
    assert img.ndim == ref.ndim
    assert img.ndim in [2,3]
    if img.ndim == 2:
        axis = (0,1)
    elif img.ndim == 3:
        axis = (1,2)
    # else not possible

    denominator = np.sum(np.real(ref * np.conj(ref)), axis=axis)
    nominator = np.sum(np.real((img - ref) * np.conj(img - ref)), axis=axis)
    rmse = np.mean(np.sqrt(nominator / denominator))
    return rmse

def ssim(img, ref, dynamic_range=None):
    """ Compute SSIM. If inputs are 3D, average over axis=0.
        If dynamic_range != None, the same given dynamic range will be used for all slices in the volume. """
    assert img.ndim == ref.ndim
    assert img.ndim in [2, 3]
    if img.ndim == 2:
        img = img[np.newaxis]
        ref = ref[np.newaxis]

    # ssim averaged over slices
    ssim_slices = []
    ref_abs = np.abs(ref)
    img_abs = np.abs(img)

    for i in range(ref_abs.shape[0]):
        if dynamic_range == None:
            drange = np.max(ref_abs[i]) - np.min(ref_abs[i])
        else:
            drange = dynamic_range
        _, ssim_i = compare_ssim(img_abs[i], ref_abs[i],
                                 data_range=drange,
                                 gaussian_weights=True,
                                 use_sample_covariance=False,
                                 full=True)
        ssim_slices.append(np.mean(ssim_i))

    return np.mean(ssim_slices)