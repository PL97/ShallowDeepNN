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

if __name__ == "__main__":
    # SVCCA MNIST Example Cross-layer Comparison, with random baseline as comparison
    # The following test example shows:
    # 1) Cross-layer subspace principle angles (cos values) of activation feature #1 and activation feature #2 (from different layers)
    # 2) Principle angles of one activation feature #1 and a random data matrix with the same size as #2
    test_case_dir = './model_activations/MNIST'
    model_idx = 0
    principle_element_threshold = 0.99
    idx_start, idx_end = 1, 5
    avg_angle_log = np.zeros([idx_end-idx_start, idx_end-idx_start])
    avg_angle_log_baseline = np.zeros_like(avg_angle_log)
    for idx_1 in range(idx_start, idx_end, 1):
        # This Loop Choice Include a trivial test on principle angles analysis with itself
        # Which theoretically should be all 1
        for idx_2 in range(idx_1, idx_end, 1):
            print('Comparing Layer %d and Layer %d' % (idx_1, idx_2))
            # Load Stored Feature Samples
            data_1_path = os.path.join(test_case_dir, 'model_%d_lay%02d.p' % (model_idx, idx_1))
            data_2_path = os.path.join(test_case_dir, 'model_%d_lay%02d.p' % (model_idx, idx_2))
            with open(data_1_path, "rb") as f:
                acts1 = pickle.load(f)
            with open(data_2_path, "rb") as f:
                acts2 = pickle.load(f)
            print("activation shapes", acts1.shape, acts2.shape)

            # Generate baseline random vectors
            # b1 = np.random.randn(*acts1.shape)
            b2 = np.random.randn(*acts2.shape)
            print('Computing Cross Layer Principle Angles...')
            res_1 = subspace_ananysis_cpu(acts1, acts2,
                                          threshold=principle_element_threshold)

            print('Computing Baseline Principle Angles...')
            res_2 = subspace_ananysis_cpu(acts1, b2,
                                          threshold=principle_element_threshold)

            # # Log avg summary of principle angles (cos value)
            # avg_angle_log[idx_1 - 1, idx_2 - 1] = np.mean(res_1)
            # avg_angle_log_baseline[idx_1-1, idx_2-1] = np.mean(res_2)

            #  Another Novel Check:
            # Since we now know only 8-directions
            # Summarizes the principle elements in the last layer
            # Lets observe how much cross-layer difference in using only 8 directions
            # For all layers
            avg_angle_log[idx_1-1, idx_2 - 1] = np.mean(res_1[0:9])
            avg_angle_log_baseline[idx_1 - 1, idx_2 - 1] = np.mean(res_2[0:9])

            # Visualization
            if idx_1 != idx_2:
                _plot_save(res_1, res_2,
                           xlabel='Principle Angles (Cos)',
                           ylabel='Value',
                           save_plot_name='Layer%d_Layer%d' % (idx_1, idx_2),
                           root_dir=test_case_dir)
    print(avg_angle_log)
    print(avg_angle_log_baseline)

    # Cross Network Comparison

