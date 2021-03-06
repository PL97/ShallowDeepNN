# This script generates features from CNN modules and save to disks

import argparse, os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from collections import Counter
import random
from scripts.utils.chexpert_dataset import preprocess, CheXpert
from scripts.models.dense_model import densenet121
from torchvision.models.densenet import _DenseBlock
import torchvision


def count_label(L):
    for i in range(L.shape[1]):
        print(Counter(L[:, i]))


def load_data(input_num, batch_size=16):
    RootPath = "/home/jusun/shared/Stanford_Dataset/"
    df = pd.read_csv(RootPath + "CheXpert-v1.0-small/TEST.csv")
    df = preprocess(df, RootPath)

    label = np.asarray(df.iloc[:, 1:]).astype(int)
    count_label(label)
    IDX = sample_data(label, input_num)
    # print(len(set(IDX)), len(IDX))
    # count_label(label[IDX])
    df_sample = (df.iloc[IDX, :]).reset_index(drop=True)
    test_dl = DataLoader(CheXpert(df_sample), batch_size=batch_size, shuffle=True)  # modify batch size here
    return test_dl


def sample_data(label, input_num, kept_idx=[]):
    if input_num <= 0:
        return []
    if input_num < label.shape[1]:
        tmp_idx = list(set(range(label.shape[0])) - set(kept_idx))
        random.shuffle(tmp_idx)
        return tmp_idx[:input_num]

    IDX = []
    N = int(input_num / (label.shape[1]))
    for i in range(label.shape[1]):
        tmp = label[:, i]
        idx = np.where(tmp == 1)[0]
        idx = list(set(idx) - set(IDX) - set(kept_idx))
        random.shuffle(idx)
        num_of_interest = max(0, N - np.sum(label[IDX, i]))
        IDX.extend(idx[:num_of_interest])

    if len(IDX) == 0:
        tmp_idx = list(set(range(label.shape[0])) - set(kept_idx))
        random.shuffle(tmp_idx)
        return tmp_idx[:input_num]

    # print("input_num:", input_num, len(IDX))
    # print(IDX)
    if len(IDX) < input_num:
        # print(label.shape, input_num, len(IDX))
        return IDX + sample_data(label, input_num - len(IDX), kept_idx=kept_idx + IDX)
    return IDX


# if __name__ == "__main__":
#     df = load_data(13)
#     for d, l in df:
#         print(d.shape, l)

def to_input_tensor(np_array, device, requires_grad=False):
    tensor_out = torch.from_numpy(np_array)
    tensor_out = tensor_out.to(device, dtype=torch.float32)
    tensor_out.requires_grad = requires_grad
    return tensor_out


def main(args):
    # ==== Configs ====
    module_idx = args.layer_idx

    use_model = args.use_model  # Specified the trained model type
    model_idx = args.model_idx  # The idx of the picked model within a use_model type.
    batch_size = args.batch_size  # Batch Size
    data_idx = 0

    # ==== Other Automatic Settings ====
    activation_root_dir = os.path.abspath('model_activations')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load Data
    df = load_data(input_num=2000,
                   batch_size=batch_size)
    data_len = len(df)
    print('data_length: ', data_len)

    if use_model == 'Dense121':
        model = densenet121(num_classes=13, pretrained=False)
        model_path = 'saved_models/dense121/dense121.pth.tar'
        # Load Model Params
        state_dict = torch.load(model_path)['state_dict']
        key_list = state_dict.keys()

        for key, param in model.state_dict().items():
            if 'module.' + key in key_list:
                model.state_dict()[key].copy_(state_dict['module.' + key])
            else:
                print('Did not find weights for: ', key)

        model_type = 'dense'
        splict_block = _DenseBlock
        activation_root_dir = os.path.join(activation_root_dir, use_model)
        if not os.path.exists(activation_root_dir):
            os.mkdir(activation_root_dir)
    else:
        print('Unsupport use_model type yet. Check input param!')

    model.to(device)
    model.eval()
    if model_type == 'dense':
        module_lst = list(list(model.children())[0].children())
    else:
        print('Unsupport use_model type yet. Check input param!')

    # if type(module_lst[module_idx]) == splict_block and module_idx > 1:
    if type(module_lst[module_idx]) == splict_block:
        print(' ===> Computing %d Layer Activations ....' % module_idx)
        if model_idx < 10:
            test_module = torch.nn.Sequential(*module_lst[0:module_idx+2])
        elif model_idx == 10:
            test_module = torch.nn.Sequential(*module_lst[:])
        else:
            print('Module Idx Exceed Model Structure.')
            return None

        activation_log = []
        partition_num, splict_batch_num = 0, 512

        for tensor_input, _ in df:
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
    parser.add_argument('--batch_size', dest='batch_size', type=int, action='store', default=8)
    parser.add_argument('--use_model', dest='use_model', type=str, action='store', default='Dense121')
    parser.add_argument('--model_idx', dest='model_idx', type=int, action='store', default=0)
    parser.add_argument('--layer_idx', dest='layer_idx', type=int, action='store', default=10)

    args = parser.parse_args()
    main(args)
    # model_path = 'saved_models/dense121/dense121.pth.tar'
    # model = densenet121(num_classes=13, pretrained=False)
    # state_dict = torch.load(model_path)['state_dict']
    # key_list = state_dict.keys()
    #
    # for key, param in model.state_dict().items():
    #     if 'module.' + key in key_list:
    #         model.state_dict()[key].copy_(state_dict['module.' + key])
    #     else:
    #         print('Did not find weights for: ', key)
    #
    # a = list(list(model.children())[0].children())
    # for block in a:
    #     print('Block type: ', type(block))
    #     print(type(block) is _DenseBlock)
    # print()

