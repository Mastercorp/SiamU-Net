from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import rotate


def flip_lr(img):
    """Flip an PIL.Image left to right"""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def flip_tb(img):
    """Flip an PIL.Image top to bottom"""
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def randscalledtrans(img, scalex, scaley, sigma):
    aimg = np.asarray(img)
    dx = np.random.uniform(-1, 1, size=aimg.shape)
    dy = np.random.uniform(-1, 1, size=aimg.shape)
    dx /= np.max(np.abs(dx))
    dy /= np.max(np.abs(dy))

    dx_gaus = gaussian_filter(dx, sigma, order=0, mode='mirror', truncate=3) * scalex
    dy_gaus = gaussian_filter(dy, sigma, order=0, mode='mirror', truncate=3) * scaley
    row_norm = np.linalg.norm([np.reshape(dx_gaus, (-1, 1)), np.reshape(dy_gaus, (-1, 1))], axis=0)
    dx_gaus_norm = np.reshape(dx_gaus, (-1, 1)) / row_norm * scalex
    dy_gaus_norm = np.reshape(dy_gaus, (-1, 1)) / row_norm * scaley
    x, y = np.meshgrid(np.arange(aimg.shape[1]), np.arange(aimg.shape[0]), indexing='xy')

    indices = [np.reshape(y, (-1, 1)) + dy_gaus_norm, np.reshape(x, (-1, 1)) + dx_gaus_norm]

    img = Image.fromarray(
        map_coordinates(aimg, indices, order=1, mode='mirror').reshape((aimg.shape[0], aimg.shape[1])))

    return img, indices


def randtrans(img, scalex, scaley, sigma):
    aimg = np.asarray(img)
    dx = np.random.uniform(-1, 1, size=aimg.shape)
    dy = np.random.uniform(-1, 1, size=aimg.shape)
    dx /= np.max(np.abs(dx))
    dy /= np.max(np.abs(dy))

    dx_gaus = gaussian_filter(dx, sigma, order=0, mode='mirror', truncate=3) * scalex
    dy_gaus = gaussian_filter(dy, sigma, order=0, mode='mirror', truncate=3) * scaley
    x, y = np.meshgrid(np.arange(aimg.shape[1]), np.arange(aimg.shape[0]), indexing='xy')

    indices = [np.reshape(y + dy_gaus, (-1, 1)), np.reshape(x + dx_gaus, (-1, 1))]

    img = Image.fromarray(
        map_coordinates(aimg, indices, order=1, mode='mirror').reshape((aimg.shape[0], aimg.shape[1])))

    return img, indices


def gradtrans(img, scalex, scaley, sigma):
    """uses the gradient and scaling values of img to transform the image

    Args:
        img (PIL.Image):  image
        scalex (int): scaling factor for x direction for gradient impact
        scaley (int): scaling factor for y direction for gradient impact
        sigma (int):  sigma value for gauss filter

    Returns:
        img (PIL.Image), indices (list): transformed img
        Care! A label can contain values between 0 and 255, caused by the transformation.
        indices is a list, which contains the mapping coordinates.


    """

    aimg = np.asarray(img)

    dy = np.gradient(aimg.astype('float64'), axis=0)
    dx = np.gradient(aimg.astype('float64'), axis=1)

    # divide by max abs value of whole matrix
    dx /= np.max(np.abs(dx))
    dy /= np.max(np.abs(dy))

    dx_gaus = gaussian_filter(dx, sigma, order=0, mode='mirror', truncate=3) * scalex
    dy_gaus = gaussian_filter(dy, sigma, order=0, mode='mirror', truncate=3) * scaley

    x, y = np.meshgrid(np.arange(aimg.shape[1]), np.arange(aimg.shape[0]), indexing='xy')

    indices = [np.reshape(y + dy_gaus, (-1, 1)), np.reshape(x + dx_gaus, (-1, 1))]

    img = Image.fromarray(
        map_coordinates(aimg, indices, order=1, mode='mirror').reshape((aimg.shape[0], aimg.shape[1])))

    return img, indices


def indextrans(img, indices):
    """map the img and label to the given indices.

    Args:
        img (PIL.Image):  image
        indices (list): the coordinates at which the image is mapped


    Returns:
        img (PIL.Image) : transformed img
        Care! a label img can contain values between 0 and 255, caused by the transformation.
        Use maplabel to handle errors.


    """
    aimg = np.asarray(img)

    img = Image.fromarray(
        map_coordinates(aimg, indices, order=1, mode='mirror').reshape((aimg.shape[0], aimg.shape[1])))

    return img


def maplabel(label, threshold, labelmap1, labelmap2):
    """maps values from an image to label1 and label2, depending on the threshold

    Args:
        label (PIL.Image): label image
        threshold (int): threshold value
        labelmap1 (int): mapping value
        labelmap2 (int): mapping value

    Returns:
        label (PIL.Image): label image with only 2 values, labelmap1 and labelmap2

    """

    # use of array instead of asarray, to create a copy which can be edited.
    label = np.array(label)
    low_values_flag = label <= threshold
    high_values_flag = label > threshold
    label[low_values_flag] = labelmap1
    label[high_values_flag] = labelmap2
    label = Image.fromarray((label).astype(np.uint8))

    return label


def combine_img(img1, img2):
    """add img2 as second channel to img2, returns np.array"""
    aimg1 = np.asarray(img1)
    aimg2 = np.asarray(img2)
    img = np.dstack((aimg1, aimg2))
    return img


def rotate_img(img, angle):
    """rotate given img at a given angle. If label: dont forget to remap values with maplabel

    Args:
        img (PIL.Image):  image
        angle (int): angle in grad
    Returns:
        imgrot (PIL.Image): rotated image

    """

    imgrot = np.asarray(img)
    imgrot = rotate(imgrot, angle, mode='reflect')
    imgrot = Image.fromarray((imgrot * 255).astype(np.uint8))
    return imgrot
