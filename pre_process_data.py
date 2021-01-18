# To Le: I'm just trying to use this script to down-sample the Kaggle dataset
# Hopefully it won't use a lot of resources and speedup calculating the interim features

import os
import numpy as np
import torch
from torchvision import transforms
import argparse

data_transforms = torch.nn.Sequential(
    transforms.Resize(size=[224, 224]),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)
scripted_transforms = torch.jit.script(data_transforms)


def to_input_tensor(np_array, device, requires_grad=False):
    tensor_out = torch.from_numpy(np.transpose(np_array, [0, 3, 1, 2]))
    tensor_out = scripted_transforms(tensor_out)
    tensor_out = tensor_out.to(device, dtype=torch.float32)
    tensor_out.requires_grad = requires_grad
    return tensor_out


def main(args):
    # ==== Configs ====
    data_idx = args.data_idx
    batch_size = args.batch_size
    print('Input to argparser: ')
    print(data_idx, batch_size)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load Data
    data_path = os.path.abspath('dataset/Retina_kaggle/diabetic-retinopathy-%d.npz' % data_idx)
    DATA_RAW = np.load(data_path)["data"][0:16, :, :, :]
    data_len = DATA_RAW.shape[0]
    print('data_length: ', data_len)
    print('Data Mat Shape: ', DATA_RAW.shape)

    save_path = os.path.abspath('dataset/Retina_kaggle/downsampled')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_file_path = os.path.join(save_path,
                                  data_path[-17:-4])

    downsampled_log = []
    n_batch = data_len // batch_size
    for batch_idx in range(n_batch):
        idx_start = batch_idx * batch_size
        idx_end = batch_idx * batch_size + batch_size

        data_numpy = DATA_RAW[idx_start:idx_end, :, :, :] / 255
        tensor_input = to_input_tensor(data_numpy,
                                       device=device,
                                       requires_grad=False)
        downsampled = tensor_input.detach().cpu().numpy()
        downsampled_log.append(downsampled)
    downsampled_log = np.concatenate(downsampled_log, axis=0)
    np.save(save_file_path, downsampled_log)
    print('Check saved array size: ', downsampled_log.shape)
    print('File Saved.')


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Specify 'data idx' and 'process batch size'.")

    parser.add_argument('--data_idx', dest='data_idx', type=int, action='store', default=0)
    parser.add_argument('--batch_size', dest='batch_size', type=int, action='store', default=16)

    args = parser.parse_args()
    main(args)


