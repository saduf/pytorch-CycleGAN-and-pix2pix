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

from models.networks import define_G, define_E, get_gating_heads


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

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
    os.remove('tmp.pt')


if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--quantized', action='store_true', help='Quantize model weights')
    args = vars(ap.parse_args())
    quantized = args["quantized"]

    print("Quantized: {}".format(quantized))

    conversion_start = time.time()
    # Initialize Generator
    gen_filename = "checkpoints/noisy2clean/review2/epoch5/latest_net_G_A.pth"
    state_dict = torch.load(gen_filename)
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
    if quantized:
        print("Generator model size:")
        print_model_size(generator_model)
        generator_model_q = torch.quantization.quantize_dynamic(
            generator_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
        print("Generator model quantized size:")
        print_model_size(generator_model_q)

    # generator_script = torch.jit.trace(generator_model, generator_input)
    generator_script = torch.jit.script(generator_model)
    gen_script_file = os.path.splitext(gen_filename)[0] + '_s.pt'
    print("saving: {}".format(gen_script_file))
    torch.jit.save(generator_script, gen_script_file)
    # print(generator_script.code)

    # Initialize Embedder
    e_filepath = "checkpoints/noisy2clean/review2/epoch5/latest_net_E.pth"
    state_dict_e = torch.load(e_filepath)
    if hasattr(state_dict_e, '_metadata'):
        del state_dict_e._metadata
    netE = define_E(input_nc=1, classes=5, init_type="normal", init_gain=0.02, gpu_ids=[])
    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict_e.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict_e, netE, key.split('.'))
    netE.load_state_dict(state_dict_e)
    netE.eval()
    if quantized:
        print("Embedder model size:")
        print_model_size(netE)
        netE_q = torch.quantization.quantize_dynamic(
            netE, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )
        print("Embedder model quantized size")
        print_model_size(netE_q)
    embedder_input = torch.rand(1, 1, 256, 256)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    embedder_script = torch.jit.trace(netE, embedder_input)
    # embedder_script = torch.jit.script(netE)
    e_script_file = os.path.splitext(e_filepath)[0] + '_s.pt'
    print("saving: {}".format(e_script_file))
    torch.jit.save(embedder_script, e_script_file)
    # print(embedder_script.graph_for(embedder_input))


    # Initialize Gating Network
    chkp_dir = "checkpoints/noisy2clean/review2/epoch5"  # save all the checkpoints to save_dir
    netGN = get_gating_heads(head_count=9, input_nc=64, output_nc=256, init_type="normal",
                             init_gain=0.02, gpu_ids=[])
    load_gating_ckpt(netGN, "latest", "GN_A", chkp_dir, gpu_ids=[])
    gn_input = torch.rand(1, 1, 512, 64)
    for idx, gate_heads in enumerate(netGN):
        g1, g2 = gate_heads
        if quantized:
            print("Gate 1 model size:")
            print_model_size(g1)
            g1 = torch.quantization.quantize_dynamic(
                g1, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
            )
            print("Gate 1 model quantized size")
            print_model_size(g1)
            print("Gate 2 model size:")
            print_model_size(g2)
            g2 = torch.quantization.quantize_dynamic(
                g2, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
            )
            print("Gate 1 model quantized size")
            print_model_size(g2)
        g1_script = torch.jit.trace(g1, gn_input)
        g2_script = torch.jit.trace(g2, gn_input)
        g1_script_filename = chkp_dir + os.path.sep + "gates" + os.path.sep + 'g1_' + str("GN_A") + '_' + str(idx) + ".pt"
        g2_script_filename = chkp_dir + os.path.sep + "gates" + os.path.sep + 'g2_' + str("GN_A") + '_' + str(idx) + ".pt"
        print("saving: {}".format(g1_script_filename))
        torch.jit.save(g1_script, g1_script_filename)
        # print(g1_script.graph_for(gn_input))
        print("saving: {}".format(g2_script_filename))
        torch.jit.save(g2_script, g2_script_filename)
        # print(g2_script.graph_for(gn_input))

    print("Export model to Torchscript time: {}".format(time.time() - conversion_start))


