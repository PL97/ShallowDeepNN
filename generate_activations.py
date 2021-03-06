# This script generates features from CNN modules and save to disks
# The saved features (np.array) are thus ready for feature analysis
# E.g., subspace principle angle analysis / SVCCA

import os
from scripts.models.model_cbr import CBR_Tiny, CBR
import numpy as np
import torch
import argparse


def to_input_tensor(np_array, device, requires_grad=False):
    tensor_out = torch.from_numpy(np_array)
    tensor_out = tensor_out.to(device, dtype=torch.float32)
    tensor_out.requires_grad = requires_grad
    return tensor_out


def main(args):
    # ==== Configs ====
    data_idx = args.data_idx  # in list(range(10)), help pick the saved input data file
    module_idx = args.layer_idx

    use_model = args.use_model  # Specified the trained model type
    model_idx = args.model_idx  # The idx of the picked model within a use_model type.

    batch_size = args.batch_size  # Batch Size
    data_set = args.data_set

    # ==== Other Automatic Settings ====
    activation_root_dir = os.path.abspath('model_activations')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load Data
    data_path = os.path.abspath(data_set)
    data_path = os.path.join(data_path, 'retinopathy-%d.npy' % data_idx)
    DATA_RAW = np.load(data_path)
    data_len = DATA_RAW.shape[0]
    print('data_length: ', data_len)
    print('Data Mat Shape: ', DATA_RAW.shape)

    if use_model == 'CBR_Tiny':
        model = CBR_Tiny()
        model_path = os.path.abspath('saved_models/CBR/CBR_Tiny.pt')
        model.load_state_dict(torch.load(model_path))
        model_type = 'CBR'
        splict_block = CBR
        activation_root_dir = os.path.join(activation_root_dir, use_model)
        if not os.path.exists(activation_root_dir):
            os.mkdir(activation_root_dir)
    else:
        print('Unsupport use_model type yet. Check input param!')

    model.to(device)
    model.eval()
    if model_type == 'CBR':
        module_lst = list(model.children())
    else:
        print('Unsupport use_model type yet. Check input param!')


    # if type(module_lst[module_idx]) == splict_block and module_idx > 1:
    if type(module_lst[module_idx]) == splict_block:
        print(' ===> Computing %d Layer Activations ....' % module_idx)
        test_module = torch.nn.Sequential(*module_lst[0:module_idx+1])

        activation_log = []
        partition_num, splict_batch_num = 0, 512
        n_batch = data_len // batch_size
        for batch_idx in range(n_batch):
            idx_start = batch_idx * batch_size
            idx_end = batch_idx * batch_size + batch_size

            data_numpy = DATA_RAW[idx_start:idx_end, :, :, :]
            tensor_input = to_input_tensor(data_numpy,
                                           device=device,
                                           requires_grad=False)
            activation = test_module(tensor_input).detach().cpu().numpy()
            activation_log.append(activation)
            if len(activation_log) >= splict_batch_num / batch_size:
                # Check activation dimensions
                activation_log = np.concatenate(activation_log, axis=0)
                save_file_path = os.path.join(activation_root_dir,
                                              'ds%d_model%d_lay%2d_part%d.npy' % (data_idx,
                                                                                  model_idx,
                                                                                  module_idx,
                                                                                  partition_num))
                np.save(save_file_path, activation_log)
                partition_num += 1
                activation_log = []
        # Check activation dimensions
        if len(activation_log) >= 2:
            activation_log = np.concatenate(activation_log, axis=0)
            save_file_path = os.path.join(activation_root_dir,
                                          'ds%d_model%d_lay%2d_part%d.npy' % (data_idx,
                                                                              model_idx,
                                                                              module_idx,
                                                                              partition_num))
            np.save(save_file_path, activation_log)
        print('Activation Calculation Completed. Save File.')


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Specify 'data idx' and 'process batch size'.")

    parser.add_argument('--data_idx', dest='data_idx', type=int, action='store', default=0)
    parser.add_argument('--layer_idx', dest='layer_idx', type=int, action='store', default=6)
    parser.add_argument('--batch_size', dest='batch_size', type=int, action='store', default=16)
    parser.add_argument('--use_mode', dest='use_model', type=str, action='store', default='CBR_Tiny')
    parser.add_argument('--model_idx', dest='model_idx', type=int, action='store', default=0)
    parser.add_argument('--data_set', dest='data_set', type=str, action='store',
                        default='dataset/Retina_kaggle/downsampled')

    args = parser.parse_args()
    main(args)


