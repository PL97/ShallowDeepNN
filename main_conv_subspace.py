import os
import argparse
import numpy as np
import scipy.linalg as sl
import numpy.linalg as nlg
from matplotlib import pyplot as plt


def _plot_save(arr_1, arr_baseline, xlabel, ylabel, save_plot_name, root_dir):
    save_dir = os.path.join(root_dir, 'visualization')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file_name = os.path.join(save_dir, '%s.jpg' % save_plot_name)
    arr_1_avg = np.mean(arr_1)
    arr_baseline_avg = np.mean(arr_baseline)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(arr_1, lw=2.0, label='%s-avg: %f' % (save_plot_name, arr_1_avg))
    ax1.plot(arr_baseline, lw=2.0, label='rand_baseline-avg: %f' % arr_baseline_avg)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid()
    plt.savefig(save_file_name, format='jpg')


def subspace_ananysis_cpu(data_mat_1, data_mat_2, verbose=False):
    """
    Args:
        data_mat_1: np.array --- (n_feature, n_data) --- array already reduced dimension
        data_mat_2: np.array --- (n_feature, n_data)
        verbose:  True if you want to see a lot of prints.

    Returns:

    """
    # Compute QR Decomposition
    Q_x, R_x = nlg.qr(data_mat_1.T)
    Q_y, R_y = nlg.qr(data_mat_2.T)
    print('Q_x, Q_y: ', Q_x.shape, Q_y.shape)
    Q_x = Q_x.T
    Q_y = Q_y.T

    # Form C with SVD
    C = np.dot(Q_x, Q_y.T)
    U, cos_thetas, V_transpose = sl.svd(C, full_matrices=False)  # U -- (p, r), V -- (q, r) where r = min(p, q)
    print('Check cos_theta dim: ', cos_thetas.shape)

    # Principle Vectors in step 3 may not be so interested right now.
    return cos_thetas


def analysis_data(args):
    root_dir = args.activation_dir
    model_idx_1 = args.model_1
    model_idx_2 = args.model_2
    layer_idx_1 = args.layer_1
    layer_idx_2 = args.layer_2

    svd_act_dir = os.path.join(root_dir, 'svd_activations')
    if not os.path.exists(svd_act_dir):
        os.mkdir(svd_act_dir)
    data_1_dir = os.path.join(svd_act_dir, 'model%d_lay%d.npy' % (model_idx_1, layer_idx_1))
    data_2_dir = os.path.join(svd_act_dir, 'model%d_lay%d.npy' % (model_idx_2, layer_idx_2))

    if os.path.isfile(data_1_dir) and os.path.isfile(data_2_dir):
        data_1 = np.load(data_1_dir)
        data_2 = np.load(data_2_dir)
        res = subspace_ananysis_cpu(data_1, data_2, verbose=False)
        return res, data_1.shape, data_2.shape
    else:
        print('No such activation stored.')
        return None, None, None


def analysis_benchmark(shape_1, shape_2):
    data_1 = np.random.random_sample(shape_1)
    data_2 = np.random.random_sample(shape_2)
    res = subspace_ananysis_cpu(data_1, data_2, verbose=False)
    return res


def main(args):
    m1 = args.model_1
    m2 = args.model_2
    lay_1 = args.layer_1
    lay_2 = args.layer_2

    res, shape_1, shape_2 = analysis_data(args)
    if res is not None:
        res2 = analysis_benchmark(shape_1, shape_2)

        save_plot_name = 'model%d_lay%d_model%d_lay%d' % (m1, lay_1,
                                                          m2, lay_2)
        _plot_save(res, res2,
                   xlabel='Principle Angles (Cos)',
                   ylabel='values',
                   save_plot_name=save_plot_name,
                   root_dir=args.activation_dir)
    else:
        print('Activation specification wrong. Check them.')


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Specify 'data idx' and 'process batch size'.")

    parser.add_argument('--activation_dir', dest='activation_dir', type=str, action='store',
                        default='model_activations/CBR_Tiny')
    parser.add_argument('--model_1', dest='model_1', type=int, action='store', default=0)
    parser.add_argument('--model_2', dest='model_2', type=int, action='store', default=0)
    parser.add_argument('--layer_1', dest='layer_1', type=int, action='store', default=4)
    parser.add_argument('--layer_2', dest='layer_2', type=int, action='store', default=6)

    args = parser.parse_args()
    main(args)

