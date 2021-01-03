import os
import torch
from torchvision.models import resnet50
from matplotlib import pyplot as plt
import time
import numpy as np
from scripts.utils import cca_core
from scipy import interpolate


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def compute_features_resnet(dataset_dir,
                            model_dir,
                            save_dir,
                            batch_size=8,
                            test_data_len=128):
    """
    Computes the block-wise (split by bottle-neck module)intermediate features of a pretrained resnet.
    The intermediate features are stored as ndarray to the path specified by 'save_dir'

    Args:
        dataset_dir: Directory of data.npy (only input arrays).
        model_dir: Directory of the pytorch pretrained model, in .pt format, containing only state_dict().
        save_dir: destination path to store the intermediate features.
        batch_size:
        test_data_len: number of testing images used to analyse the network.
    Returns:

    """
    n_batch = test_data_len // batch_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Preparing Testing data
    DATA = np.load(os.path.join(dataset_dir, 'data.npy'))[0:test_data_len, :, :, :]
    DATA = np.transpose(DATA, [0, 3, 1, 2])
    # Dataset Normalisation (Should be consistent with training stage)
    mean_transform = [0.485, 0.456, 0.406]
    std_transform = [0.229, 0.224, 0.225]
    for i in range(len(mean_transform)):
        DATA[:, i, :, :] = (DATA[:, i, :, :] - mean_transform[i]) / std_transform[i]

    model = resnet50(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    model.to(device)
    module_lst = list(model.children())

    for idx in range(len(module_lst)):
        if type(module_lst[idx]) == torch.nn.Sequential:
            test_module = torch.nn.Sequential(*module_lst[0:idx+1])
            test_module.eval()

            outputs = []
            for batch_idx in range(n_batch):
                inputs = DATA[batch_idx*batch_size:batch_idx*batch_size+batch_size,
                              :, :, :]
                inputs = torch.from_numpy(inputs).to(device, dtype=torch.float32)
                features = test_module(inputs).detach().cpu().data.numpy()
                outputs.append(features)
            outputs = np.concatenate(outputs, axis=0)
            print(outputs.shape)
            save_file_name = os.path.join(save_dir, 'Block%d.npy' % idx)
            np.save(save_file_name, outputs)
        else:
            pass


def cca_analysis(array_1, array_2):
    """
    According to SVCCA paper ---
        CCA analysis can have different channel number, but should always have the same number of data points
    Intuition:
        Analogously thinking a CNN block as 'channel' number of 2d (h*w) neurons. Thus each 2d intermeaddiate
        feature are the activations of different number of neurons.
    Method:
        Using interpolation (Linear) to interpolate the smaller filter into the larger ones

    Args:
        array_1:
        array_2:

    Returns:

    """
    num_data_1, num_filter_1, h1, w1 = array_1.shape
    num_data_2, num_filter_2, h2, w2 = array_2.shape
    # *** Note: array_1 and array_2 are assumed to be h1<=h2, w1<=w2 ***
    # *** num_data_1 = num_data_2 ***
    array_interp = np.zeros((num_data_1, num_filter_1, h2, w2))
    for d in range(num_data_1):
        for c in range(num_filter_1):
            # Formulate Interpolation Function
            idxs1 = np.linspace(0, h1, h1, endpoint=False)
            idxs2 = np.linspace(0, w1, w1, endpoint=False)
            arr = array_1[d, c, :, :]
            f_interp = interpolate.interp2d(idxs1, idxs2, arr, kind='linear')
            # Create Larger Arrangement
            larger_idxs1 = np.linspace(0, h1, h2, endpoint=False)
            larger_idxs2 = np.linspace(0, w1, w2, endpoint=False)
            array_interp[d, c, :, :] = f_interp(larger_idxs1, larger_idxs2)
            # # === Debug plot for interpolation check === # #
            # f_1 = plt.figure()
            # ax_1 = f_1.add_subplot(1,2,1)
            # ax_1.imshow(arr)
            # ax_2 = f_1.add_subplot(1,2,2)
            # ax_2.imshow(array_interp[d,c,:,:])
            # plt.show()
            # plt.close(f_1)
            # # ========================================= # #
    array_1 = np.transpose(array_interp, [0, 2, 3, 1])
    array_2 = np.transpose(array_2, [0, 2, 3, 1])

    array_1 = array_1.reshape(-1, num_filter_1)
    array_2 = array_2.reshape(-1, num_filter_2)
    print('Data shape: ', array_1.shape)
    start_time = time.time()
    f_results = cca_core.robust_cca_similarity(array_1.T, array_2.T, epsilon=1e-10)
    # f_results = cca_core.get_cca_similarity(f_acts_1.T, f_acts_2.T, epsilon=1e-10, verbose=False)
    print('Time: {:.2f} seconds'.format(time.time() - start_time))
    _plot_helper(f_results["cca_coef1"], "CCA Coef idx", "CCA coef value")
    print(f_results["cca_coef1"])
    return f_results["cca_coef1"]


if __name__ == "__main__":
    # ====== Compute Intermediate Features ========= # #
    print('Testing SVCCA analysis')
    DATA_DIR = os.path.abspath('dataset/Messidor1')
    MODEL_DIR = os.path.abspath('saved_models/resnet50.pt')

    save_feature_dir = os.path.abspath('model_features')
    if not os.path.exists(save_feature_dir):
        os.mkdir(save_feature_dir)
    res_feature_dir = os.path.join(save_feature_dir, 'resnet')
    if not os.path.exists(res_feature_dir):
        os.mkdir(res_feature_dir)
    # The following transforms should be consistent with the training stage
    test_data_len = 128  # Modify this if you want larger dataset to compute the features
    compute_features_resnet(dataset_dir=DATA_DIR,
                            model_dir=MODEL_DIR,
                            save_dir=res_feature_dir,
                            batch_size=8,
                            test_data_len=test_data_len)
    # ================================================ # #


    # ===== CCA Analysis ====
    feature_dir = os.path.abspath('model_features/resnet')

    #  Modify this based on the model saved feature above
    idx_start, idx_end = 4, 8

    # Data_length depending on the hardware computation ability
    data_length = 16  # Ideally the larger the better
    res_storage = np.eye((idx_end-idx_start), dtype=float)
    for i in range(idx_start, idx_end, 1):
        for j in range(i+1, idx_end, 1):
            array_1 = np.load(os.path.join(feature_dir, 'Block%d.npy' % i))[0:data_length, :, :, :]
            array_2 = np.load(os.path.join(feature_dir, 'Block%d.npy' % j))[0:data_length, :, :, :]
            if array_1.shape[2] < array_2.shape[2]:
                res = cca_analysis(array_1, array_2)
            else:
                res = cca_analysis(array_2, array_1)
            res_storage[i-idx_start, j-idx_start] = res
    cca_res_dir = os.path.join(feature_dir, 'cca_res.npy')
    print(res_storage)
    np.save(cca_res_dir, res_storage)



