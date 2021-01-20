import os
import argparse
import numpy as np
import scripts.utils.subspace_analysis as sa
from scripts.utils.subspace_analysis import subspace_ananysis_cpu, _plot_save


def compute_svd_activations(args):
    model_idx = args.model_idx
    layer_idx = args.layer_idx
    activation_dir = args.activation_dir

    svd_activations_dir = os.path.join(activation_dir, 'svd_activations')
    if not os.path.exists(svd_activations_dir):
        os.mkdir(svd_activations_dir)

    act_file_list = [f for f in os.listdir(activation_dir) if
                     (os.path.isfile(os.path.join(activation_dir, f)) and ('model%d_lay %d' % (model_idx, layer_idx) in f))]
    print(len(act_file_list))
    if len(act_file_list) < 1:
        print('NO such activation stored: model%d_lay %d' % (model_idx, layer_idx))
        return

    data_log = []
    for file in act_file_list:
        print('Load data: ', file)
        data_load = np.load(os.path.join(activation_dir, file))
        print('Dimension ', data_load.shape)
        data_log.append(data_load)
    data_log = np.concatenate(data_log, axis=0)
    print('All features dimension: ', data_log.shape)
    data_log = np.transpose(data_log, [1, 2, 3, 0])

    if len(data_log.shape) > 2:
        x, y, z, n_data = data_log.shape
        data_log = np.reshape(data_log, [x*y*z, n_data])
    # This is just for debugging purpose
    # data_log = data_log[0:25, :]
    # ===================================
    data_log = sa._zero_mean_vec(data_log, verbose=False)
    U1, s1, V1 = sa._svd_cpu(data_log, verbose=False)
    keep_dim = sa._find_dimension_cpu(s1, threshold=0.99, verbose=False)

    data_final = np.dot(np.diag(s1[:keep_dim]), V1[:keep_dim, :])
    save_file_name = os.path.join(svd_activations_dir,
                                  'model%d_lay%d.npy' % (model_idx, layer_idx))
    np.save(save_file_name, data_final)
    print('Saved file: ', save_file_name)
    print('Dimension: ', data_final.shape)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description=" To save storage. Computes the SVD activations.")

    parser.add_argument('--model_idx', dest='model_idx', type=int, action='store', default=0)
    parser.add_argument('--layer_idx', dest='layer_idx', type=int, action='store', default=6)
    parser.add_argument('--activation_dir', dest='activation_dir', type=str, action='store',
                        default='model_activations/CBR_Tiny')

    args = parser.parse_args()
    compute_svd_activations(args)

