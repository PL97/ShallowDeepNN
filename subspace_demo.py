import os, sys
import time
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as nlg
import cupy as cp
import cupy.linalg as clg
import pickle
import pandas
import gzip
from scripts.utils.subspace_analysis import subspace_ananysis_cpu, _plot_helper, _plot_save

sys.path.append("..")


def _subspace_compare(test_data_dir, test_model_name,
                      model_idx_1, model_idx_2,
                      layer_idx_1, layer_idx_2):
    if test_model_name == 'MNIST':
        print('Yes MNIST!')
        data_1_path = os.path.join(test_data_dir, 'model_%d_lay%02d.p' % (model_idx_1, layer_idx_1))
        data_2_path = os.path.join(test_data_dir, 'model_%d_lay%02d.p' % (model_idx_2, layer_idx_2))
        with open(data_1_path, "rb") as f:
            acts1 = pickle.load(f)
        with open(data_2_path, "rb") as f:
            acts2 = pickle.load(f)
    else:
        data_1_path = os.path.join(test_data_dir, 'model_%d_lay%02d.npy' % (model_idx_1, layer_idx_1))
        data_2_path = os.path.join(test_data_dir, 'model_%d_lay%02d.npy' % (model_idx_2, layer_idx_2))
        acts1 = np.load(data_1_path)
        acts2 = np.load(data_2_path)

    print("activation shapes", acts1.shape, acts2.shape)
    print('Comparing %s Model %d Layer %d and Model %d Layer %d' % (test_model_name,
                                                                    model_idx_1, layer_idx_1,
                                                                    model_idx_2,layer_idx_2))
    # Generate baseline random vectors
    b2 = np.random.randn(*acts2.shape)
    print(' ====> Computing Cross Layer Principle Angles...')
    res_1 = subspace_ananysis_cpu(acts1, acts2,
                                  threshold=principle_element_threshold)
    print(' ====> Computing Baseline Principle Angles...')
    res_2 = subspace_ananysis_cpu(acts1, b2,
                                  threshold=principle_element_threshold)

    # #  Another Novel Check:
    # # Since we now know only 8-directions
    # # Summarizes the principle elements in the last layer
    # # Lets observe how much cross-layer difference in using only 8 directions
    # # For all layers
    # avg_angle_log[idx_1 - 1, idx_2 - 1] = np.mean(res_1[0:9])
    # avg_angle_log_baseline[idx_1 - 1, idx_2 - 1] = np.mean(res_2[0:9])
    # Visualization
    _plot_save(res_1, res_2,
               xlabel='Principle Angles (Cos)',
               ylabel='Value',
               save_plot_name='Model%dLayer%d_Model%dLayer%d' % (model_idx_1, layer_idx_1,
                                                                 model_idx_2, layer_idx_2),
               root_dir=test_data_dir)
    # Return a mean summary of the principle angles
    return np.mean(res_1), np.mean(res_2)


if __name__ == "__main__":
    # SVCCA MNIST Example Cross-layer Comparison, with random baseline as comparison
    # The following test example shows:
    # 1) Cross-layer subspace principle angles (cos values) of activation feature #1 and activation feature #2 (from different layers)
    # 2) Principle angles of one activation feature #1 and a random data matrix with the same size as #2

    # # Specify testing model (activations)
    test_model_name = 'MNIST'
    test_case_dir = './model_activations/%s' % test_model_name
    model_idx_1, model_idx_2 = 0, 0

    # # Configure Layer index specifications
    idx_start, idx_end = 0, 5
    idx_1_list = list(range(idx_start, idx_end, 1))

    # Configuring Principle Element Threshold
    principle_element_threshold = 0.99

    avg_angle_log = np.zeros([idx_end-idx_start, idx_end-idx_start])
    avg_angle_log_baseline = np.zeros_like(avg_angle_log)
    for num_1, idx_1 in enumerate(idx_1_list):
        # This Loop Choice Include a trivial test on principle angles analysis with itself
        # Which theoretically should be all 1
        for num_2, idx_2 in enumerate(idx_1_list):
            if idx_2 >= idx_1:
                avg_test, avg_baseline = _subspace_compare(test_data_dir=test_case_dir,
                                                           test_model_name=test_model_name,
                                                           model_idx_1=model_idx_1,
                                                           model_idx_2=model_idx_2,
                                                           layer_idx_1=idx_1,
                                                           layer_idx_2=idx_2)
                avg_angle_log[num_1, num_2] = avg_test
                avg_angle_log_baseline[num_1, num_2] = avg_baseline
    print('Test Subspace Analysis summary (mean): ')
    print('Test Data: ')
    print(avg_angle_log)
    print('Baseline: ')
    print(avg_angle_log_baseline)
    print('All work done. No more sleep and check.')

