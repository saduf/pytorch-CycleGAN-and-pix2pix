import os
import time
import numpy as np
from PIL import Image, ImageOps


def convert_img_to_array(img):
    return np.array(img)


def resize_image(img, to_size, method=Image.BICUBIC):
    w, h = to_size
    h = int(h)
    w = int(w)
    print('h: %d, w: %d' % (h, w))
    return img.resize((w, h), method)


def make_256_multiple(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)


def extract_patches(img, base, is_augment=False, stride=128):
    ow, oh = img.shape
    columns = int(oh / base) * 2 - 1 if is_augment else int(oh / base)
    rows = int(ow / base) * 2 - 1 if is_augment else int(ow / base)
    print('rows: %d\ncolumns: %d' % (rows, columns))

    img_patches = []

    for row in range(rows):
        row_patches = []
        for column in range(columns):
            row_range_low = int(row * (base - stride)) if is_augment else int(row * base)
            row_range_high = int((row + 2) * (base - stride)) if is_augment else int((row + 1) * base)
            column_range_low = int(column * (base - stride)) if is_augment else int(column * base)
            column_range_high = int((column + 2) * (base - stride)) if is_augment else int((column + 1) * base)
            img_patch = img[row_range_low: row_range_high, column_range_low: column_range_high]
            row_patches.append(img_patch)
        img_patches.append(row_patches)

    print('image # of patches: %d X %d' % (len(img_patches[0]), len(img_patches)))
    return img_patches


def assemble_img_patches(img_patches):
    h_patches = []
    for row in img_patches:
        arr = np.hstack(row)
        h_patches.append(arr)
    assembled_img = np.vstack(h_patches)
    return assembled_img


def convert_arr_to_img(img):
    return Image.fromarray(np.uint8(img)).convert('L')


def save_patches(img_patches, dir_name):
    filename = dir_name.split('/')[-1]
    columns, rows = len(img_patches[0]), len(img_patches)
    total_patches = columns * rows
    image_no = 1
    for idx, row in enumerate(img_patches):
        for idy, element in enumerate(row):
            patch_filename = dir_name + os.path.sep + filename + '-' + str(idx) + '-' + str(idy) + '-of-' \
            + str(rows) + 'X' + str(columns) + '.png'
            img = Image.fromarray(element)
            img.save(patch_filename)
    print('%d patches saved for image: %s' % (total_patches, filename))

