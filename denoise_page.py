"""Denoise a document page.
Read the page from the path passed as an argument.
Load the weights as models from their checkpoints.
Resize the page to the closest multiple of 256 px, and extract non-overlapping patches.
By default the script received the path for a png with the document image in grayscale.
The patches are feed to the denoising model one at a time and saved as numpy array is a list for all patches in a row.
Once all patches rows x colums have been denoised, the page is assembled, and resized to its original size.
Denoised page is saved in the same path were the original page was read.
"""

import os
import time
import torch
import argparse
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from models.networks import define_G, define_E, get_gating_heads, get_gating_outputs
from util.process_patches import extract_patches, make_256_multiple, convert_img_to_array, \
    assemble_img_patches, resize_image


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def load_gating_ckpt(gating_network, ckpt_no, model_name, save_dir, gpu_ids):
    # print('len(gating_network: {})'.format(len(gating_network)))
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device(
        'cpu')  # get device name: CPU or GPU
    for idxe, gating_head in enumerate(gating_network):
        load_path = f"{save_dir}/gates/{str(ckpt_no).zfill(6) + '_' + str(model_name) + '_' + str(idxe)}.pt"
        # print('loading the model from %s' % load_path)
        g = gating_head
        # print('len(gating_head: {})'.format(len(gating_head)))
        # print('len(g): {}'.format(len(g)))
        # for idxi, g in enumerate(gating_head):
        g1_key = 'g1_' + str(model_name) + '_' + str(idxe)
        g2_key = 'g2_' + str(model_name) + '_' + str(idxe)
        state_dict = torch.load(load_path, map_location=str(device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        if isinstance(g[0], torch.nn.DataParallel):
            g[0].module.load_state_dict(state_dict[g1_key])
            g[1].module.load_state_dict(state_dict[g2_key])
        else:
            g[0].load_state_dict(state_dict[g1_key])
            g[1].load_state_dict(state_dict[g2_key])
        g[0].eval()
        g[1].eval()


def get_transform(grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):

    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        # print("In scaling")
        image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        # print("In direct")
        image_numpy = input_image
    return image_numpy.astype(imtype)


if __name__ == '__main__':

    total_start = time.time()

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    required_arg = ap.add_argument_group('required named arguments')
    required_arg.add_argument("-ipath", "--image_path", help="path to image to be denoised", required=True)
    args = vars(ap.parse_args())
    im_path = args["image_path"]

    init_model_start = time.time()
    # Initialize Generator
    state_dict = torch.load("checkpoints/noisy2clean/review2/epoch5/latest_net_G_A.pth")
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # make sure you pass the correct parameters to the define_G method
    generator_model = define_G(input_nc=1, output_nc=1, ngf=64, netG="resnet_9blocks",
                               norm="instance", use_dropout=False, init_gain=0.02, gpu_ids=[])

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, generator_model, key.split('.'))

    generator_model.load_state_dict(state_dict)
    generator_model.eval()

    # Initialize Embedder
    state_dict_e = torch.load("checkpoints/noisy2clean/review2/epoch5/latest_net_E.pth")
    if hasattr(state_dict_e, '_metadata'):
        del state_dict_e._metadata

    netE = define_E(input_nc=1, classes=5, init_type="normal", init_gain=0.02, gpu_ids=[])

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict_e.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict_e, netE, key.split('.'))

    netE.load_state_dict(state_dict_e)
    netE.eval()

    # Initialize Gating Network
    chkp_dir = "checkpoints/noisy2clean/review2/epoch5"  # save all the checkpoints to save_dir

    netGN = get_gating_heads(head_count=9, input_nc=64, output_nc=256, init_type="normal",
                             init_gain=0.02, gpu_ids=[])

    load_gating_ckpt(netGN, "latest", "GN_A", chkp_dir, gpu_ids=[])

    print("Tiem to init model: {:.2f}".format(time.time() - init_model_start))

    get_patches_start = time.time()
    # Obtain patches from input image
    base = 256
    img = Image.open(im_path).convert('L')
    original_shape = img.size
    print("original_shape: {}".format(original_shape))

    img_256 = make_256_multiple(img, base)
    img_256_arr = convert_img_to_array(img_256)
    img_patches = extract_patches(img_256_arr, base)
    print("Time to get patches: {:.2f}".format(time.time() - get_patches_start))

    # Denoise patches 1 at a time (Is it possible to feed a batch in CPU?)
    denoise_start = time.time()
    patches_denoised = []
    transform = get_transform(grayscale=True)
    for patches_row in img_patches:
        patches_denoised_row = []
        for real_arr in patches_row:
            # print("real real_arr shape : {}".format(real_arr.shape))
            # print("real real_arr type : {}".format(type(real_arr)))
            patch_im = Image.fromarray(real_arr)
            # print("real patch_im shape : {}".format(patch_im.size))
            # print("real patch_im type : {}".format(type(patch_im)))
            real = transform(patch_im)
            real = real.reshape((1, 1, 256, 256))
            # print("real shape : {}".format(real.shape))
            # print("real type : {}".format(type(real)))
            embedder_out, fc1_output = netE(real)
            fc1_out = torch.flatten(torch.nn.functional.softmax(fc1_output, dim=1), 1)
            gating_out = get_gating_outputs([netGN], fc1_out)
            fake = generator_model(real, gating_out[0])
            # print("fake shape: {}".format(fake.shape))
            # print("fake type: {}".format(type(fake)))
            fake = tensor2im(fake)
            fake = fake.reshape(256, 256)
            # print("fake2 shape: {}".format(fake.shape))
            # print("fake2 type: {}".format(type(fake)))
            patches_denoised_row.append(fake)
        patches_denoised.append(patches_denoised_row)
    print("Time to denoise image: {:.2f}".format(time.time() - denoise_start))

    # assemble image from patches
    assemble_start = time.time()
    assembled_img = assemble_img_patches(patches_denoised)

    # Get png from array
    denoised_im = Image.fromarray(np.uint8(assembled_img)).convert('L')

    # Resize image to original dimensions
    recovered_img = resize_image(denoised_im, original_shape)

    # Finally save denoised image
    save_filename = os.path.splitext(im_path)[0] + '-denoised.png'
    recovered_img.save(save_filename)

    print("Time to assemble image: {:.2f}".format(time.time() - assemble_start))

    print("Total time: {:.2f}".format(time.time() - total_start))

