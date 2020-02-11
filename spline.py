#!/usr/bin/env python3
"""
Greg Marcil
CS280 UC Berkeley
Practice implementation of Burt and Adelson's "A Multiresolution Spline With
Application to Image Mosaics
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage
import cv2


def im_reduce(img):
    '''
    Apply gaussian filter and drop every other pixel
    '''
    filter = 1.0 / 20 * np.array([1, 5, 8, 5, 1])
    lowpass = ndimage.filters.correlate1d(img, filter, 0)
    lowpass = ndimage.filters.correlate1d(lowpass, filter, 1)
    im_reduced = lowpass[::2, ::2, ...]
    return im_reduced


def im_expand(img, template):
    '''
    Re-expand a reduced image by interpolating according to gaussian kernel
    Include template parameter to match size, easy way to avoid off by 1 errors
    re-expanding a previous layer that may have had odd or even dimension
    '''
    # y_temp, x_temp = template.shape[:2]
    # im_expanded = np.zeros((y_temp, x_temp) + template.shape[2:], img.dtype)
    im_expanded = np.zeros(template.shape, img.dtype)
    im_expanded[::2, ::2, ...] = img

    filter = 1.0 / 10 * np.array([1, 5, 8, 5, 1])
    lowpass = ndimage.filters.correlate1d(
        im_expanded, filter, 0, mode="constant")
    lowpass = ndimage.filters.correlate1d(lowpass, filter, 1, mode="constant")
    return lowpass


def preprocess_images(im1, im2):
    '''
    Equalize image sizes and ensure odd x,y dimensions
    '''
    y1, x1, _ = im1.shape
    y2, x2, _ = im2.shape
    ''' Use dimensions of smaller image '''
    x = min(x1, x2)
    y = min(y1, y2)
    im1 = trim_image(im1, y, x)
    im2 = trim_image(im2, y, x)
    im1 = np.uint32(im1)
    im2 = np.uint32(im2)
    return im1, im2


def trim_image(im, y, x):
    '''
    Trim image to (y,x) symmetrically
    '''
    y0, x0, _ = im.shape
    y_diff = max(y0 - y, 0)
    x_diff = max(x0 - x, 0)
    ''' remove odd pixels from left/top edge '''
    y_left = y_diff // 2 + y_diff % 2
    y_right = y_diff // 2
    x_left = x_diff // 2 + x_diff % 2
    x_right = x_diff // 2
    return im[y_left:y0 - y_right, x_left:x0 - x_right]


def gaussian_pyramid(image, layers=7):
    '''
    pyramid of increasingly strongly low-pass filtered images,
    shrunk 2x h and w each layer
    '''
    pyr = [image]
    temp_img = image
    for i in range(layers):
        temp_img = im_reduce(temp_img)
        pyr.append(temp_img)
    return pyr


def laplacian_pyramid(gaussian_pyramid):
    '''
    laplacian pyramid is a band-pass filter pyramid, calculated by the
    difference between subsequent gaussian pyramid layers, terminating with top
    layer of gaussian. Laplacian pyramid can be summed to give back original
    image
    '''
    pyr = []
    for i in range(len(gaussian_pyramid) - 1):
        g_k = gaussian_pyramid[i]
        g_k_plus_1 = gaussian_pyramid[i + 1]
        g_k_1_expand = im_expand(g_k_plus_1, g_k)
        laplacian = g_k - g_k_1_expand
        pyr.append(laplacian)

    pyr.append(gaussian_pyramid[-1])
    return pyr


def laplacian_collapse(pyr):
    '''
    Rejoin all levels of a laplacian pyramid. As the pyramid is a spanning set
    of band-pass filter outputs (all frequencies represented once and only
    once), joining all levels will give back the original image, modulo
    compression loss
    '''
    ''' Start with lowest pass data, top of pyramid '''
    partial_img = pyr[-1]
    for i in range(len(pyr) - 1):
        next_lowest = pyr[-2 - i]
        expanded_partial = im_expand(partial_img, next_lowest)
        partial_img = expanded_partial + next_lowest
    return partial_img


def laplacian_pyr_join(pyr1, pyr2):
    pyr = []
    for i in range(len(pyr1)):
        left = pyr1[i]
        right = pyr2[i]
        layer = np.zeros(left.shape, left.dtype)
        _, x, _ = left.shape
        if x % 2 == 1:
            ''' odd width '''
            half = x // 2
            ''' assign halves '''
            layer[:, :half, ...] = left[:, :half, ...]
            layer[:, -half:, ...] = right[:, -half:, ...]
            ''' blend center pixel '''
            ''' not sure why adding seems to work correctly here, should have to
            divide by 2 and average? But that produces bad white-out results.
            Needs some general debugging'''
            layer[:, half, ...] = (left[:, half, ...] + right[:, half, ...])
        else:
            ''' even width '''
            half = x // 2
            ''' assign halves '''
            layer[:, :half, ...] = left[:, :half, ...]
            layer[:, -half:, ...] = right[:, -half:, ...]
        pyr.append(layer)
    return pyr


def plt_imshow(img):
    np.clip(img, 0, 255, out=img)
    img = np.uint8(img)
    plt.figure()
    plt.imshow(img)
    return 0


def main():
    plt.ion()
    im1 = matplotlib.image.imread('apple.jpg')
    im2 = matplotlib.image.imread('orange.jpg')
    im1, im2 = preprocess_images(im1, im2)

    gp_1, gp_2 = [gaussian_pyramid(im) for im in [im1, im2]]
    lp_1, lp_2 = [laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
    lp_join = laplacian_pyr_join(lp_1, lp_2)
    im_join = laplacian_collapse(lp_join)

    np.clip(im_join, 0, 255, out=im_join)
    im_join = np.uint8(im_join)
    plt.imsave('orapple.jpg', im_join)
    return 0


if __name__ == '__main__':
    main()
