# This File Wraps the function needed for the subspace distance analysis

import numpy as np
import numpy.linalg as nlg
import os
from matplotlib import pyplot as plt


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def _plot_save(arr_1, arr_baseline, xlabel, ylabel, save_plot_name, root_dir):
    save_dir = os.path.join(root_dir, 'visualization')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file_name = os.path.join(save_dir, '%s.jpg' % save_plot_name)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(arr_1, lw=2.0, label='%s' % save_plot_name)
    ax1.plot(arr_baseline, lw=2.0, label='compare with rand. baseline')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid()
    plt.savefig(save_file_name, format='jpg')


def _find_dimension_cpu(singular_values, threshold=0.95, verbose=True):
    """
        This function finds the principle component dimensions with a second norm criterion.
    Args:
        singular_values: np.array - (p, ), is the array of the singular values from an SVD result.
        threshold: constant value, determining the percentage of energy we keep with the principle vectors.

    Returns:
        dim: the reduced dimension number.

    """
    energy_total = nlg.norm(singular_values)
    for dim in range(len(singular_values)):
        percentage = nlg.norm(singular_values[:dim+1]) / energy_total
        if percentage >= threshold:
            if verbose:
                print("%f information can be preserved by %d singular vectors" % (percentage, dim))
            return dim
    print('Something may go wrong but hopefully we never need to debug this.')
    return None


def _zero_mean_vec(data_mat, verbose=True):
    """
        Form zero-mean random vector.
    Args:
        data_mat: np.array --- (p, s), where p is the vector dimension, s is the number of samples collected.

    Returns:
        np.array --- (p, s), zero-meaned new data matrix
    """
    if verbose:
        print('Form zero mean data matrix...')
    new_data_mat = data_mat - np.mean(data_mat, axis=1, keepdims=True)
    return new_data_mat


def _svd_cpu(data_mat, verbose=True):
    """
        Performs SVD decomposition with cpu.
    Args:
        data_mat: np.array --- (p, s), where p is the vector dimension, s is the number of samples collected.

    Returns:
        U, s ,V --- result of a svd decomposition where U --- (p, p); s --- (p, p); V --- (p, s)
    """
    if verbose:
        print('Taking SVD...')
    u, s, v = np.linalg.svd(data_mat, full_matrices=False)
    return u, s, v


def subspace_ananysis_cpu(data_mat_1, data_mat_2, threshold=0.9, verbose=False):
    # Zero mean collected data
    data_1 = _zero_mean_vec(data_mat_1, verbose=verbose)
    data_2 = _zero_mean_vec(data_mat_2, verbose=verbose)

    # Perform SVD and Take the most significant principle directions
    U1, s1, V1 = _svd_cpu(data_1, verbose=verbose)
    U2, s2, V2 = _svd_cpu(data_2, verbose=verbose)

    keep_dim_1 = _find_dimension_cpu(s1, threshold=threshold, verbose=verbose)
    keep_dim_2 = _find_dimension_cpu(s2, threshold=threshold, verbose=verbose)

    # Project New data mat for comparasion
    data_1 = np.dot(np.diag(s1[:keep_dim_1]), V1[:keep_dim_1, :])  # shape (keep_dim_1, s)
    data_2 = np.dot(np.diag(s2[:keep_dim_2]), V2[:keep_dim_2, :])  # shape (keep_dim_2, s)
    # The following projects back to the original data direction
    # But the difference is only rotation, thus does not affect the final result.
    # data_1 = np.dot(U1[:keep_dim_1, :keep_dim_1], np.dot(np.diag(s1[:keep_dim_1]), V1[:keep_dim_1, :]))
    # data_2 = np.dot(U2[:keep_dim_2, :keep_dim_2], np.dot(np.diag(s2[:keep_dim_1]), V2[:keep_dim_2, :]))

    # Compute QR Decomposition
    Q_x, R_x = nlg.qr(data_1.T)
    Q_y, R_y = nlg.qr(data_2.T)
    Q_x = Q_x.T
    Q_y = Q_y.T

    # Form C with SVD
    C = np.dot(Q_x, Q_y.T)
    U, cos_thetas, V_transpose = nlg.svd(C, full_matrices=False)  # U -- (p, r), V -- (q, r) where r = min(p, q)
    print('Check cos_theta dim: ', cos_thetas.shape)

    # Principle Vectors in step 3 may not be so interested right now.
    return cos_thetas
