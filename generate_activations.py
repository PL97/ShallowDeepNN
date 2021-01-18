# This script generates features from CNN modules and save to disks
# The saved features (np.array) are thus ready for feature analysis
# E.g., subspace principle angle analysis / SVCCA

import os
from scripts.models.model_cbr import CBR_Tiny, CBR
import numpy as np
import torch
from torchvision import transforms

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


if __name__ == "__main__":
    # ==== Configs ====
    data_idx = 0  # in list(range(10)), help pick the saved input data file

    use_model = 'CBR_Tiny'  # Specified the trained model type
    model_idx = 0  # The idx of the picked model within a use_model type.

    batch_size = 8  # Batch Size

    # ==== Other Automatic Settings ====
    activation_root_dir = os.path.abspath('model_activations')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load Data
    data_path = os.path.abspath('dataset/Retina_kaggle/diabetic-retinopathy-%d.npz' % data_idx)
    DATA_RAW = np.load(data_path)["data"][0:18,:,:,:]
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

    for module_idx in range(len(module_lst)):
        if type(module_lst[module_idx]) == splict_block:
            print(' ===> Computing %d Layer Activations ....' % module_idx)
            test_module = torch.nn.Sequential(*module_lst[0:module_idx+1])

            activation_log = []
            n_batch = data_len // batch_size
            for batch_idx in range(n_batch):
                idx_start = batch_idx * batch_size
                idx_end = batch_idx * batch_size + batch_size

                data_numpy = DATA_RAW[idx_start:idx_end, :, :, :] / 255
                tensor_input = to_input_tensor(data_numpy,
                                               device=device,
                                               requires_grad=False)
                activation = test_module(tensor_input).detach().cpu().numpy()
                activation_log.append(activation)
            activation_log = np.concatenate(activation_log, axis=0)
            save_file_path = os.path.join(activation_root_dir,
                                          'model_%d_lay%02d.npy' % (model_idx, module_idx))
            np.save(save_file_path, activation_log)
            print('Check Activation Size: ', activation_log.shape)
            print('Activation Calculation Completed. Save File.')




