import os
import argparse
import numpy as np
from scripts.utils.subspace_analysis import subspace_ananysis_cpu, _plot_save


def _subspace_compare(test_data_dir, data_partition_num,
                      model_idx_1, model_idx_2,
                      layer_idx_1, layer_idx_2,
                      threshold=0.9):
    acts1, acts2 = [], []
    for i_part in range(data_partition_num):
        data_1_path = os.path.join(test_data_dir,
                                   'ds%d_model%d_lay %d.npy' % (i_part,
                                                                model_idx_1,
                                                                layer_idx_1))
        data_2_path = os.path.join(test_data_dir,
                                   'ds%d_model%d_lay %d.npy' % (i_part,
                                                                model_idx_2,
                                                                layer_idx_2))
        data_1 = np.load(data_1_path)
        data_2 = np.load(data_2_path)
        acts1.append(data_1)
        acts2.append(data_2)
    acts1 = np.concatenate(acts1, axis=0)
    acts2 = np.concatenate(acts2, axis=0)
    acts1 = np.transpose(acts1, [1, 2, 3, 0])
    acts2 = np.transpose(acts2, [1, 2, 3, 0])
    print("activation shapes", acts1.shape, acts2.shape)
    print('Comparing Model %d Layer %d and Model %d Layer %d' % (model_idx_1, layer_idx_1,
                                                                 model_idx_2, layer_idx_2))
    # Generate baseline random vectors
    b2 = np.random.randn(*acts2.shape)
    print(' ====> Computing Cross Layer Principle Angles...')
    res_1 = subspace_ananysis_cpu(acts1, acts2,
                                  threshold=threshold)
    print(' ====> Computing Baseline Principle Angles...')
    res_2 = subspace_ananysis_cpu(acts1, b2,
                                  threshold=threshold)

    #  Another Novel Check:
    # Since we now know only 8-directions
    # Summarizes the principle elements in the last layer
    # Lets observe how much cross-layer difference in using only 8 directions
    # For all layers
    # avg_angle_log[idx_1 - 1, idx_2 - 1] = np.mean(res_1[0:6])
    # avg_angle_log_baseline[idx_1 - 1, idx_2 - 1] = np.mean(res_2[0:6])
    # Visualization
    _plot_save(res_1, res_2,
               xlabel='Principle Angles (Cos)',
               ylabel='Value',
               save_plot_name='Model%dLayer%d_Model%dLayer%d' % (model_idx_1, layer_idx_1,
                                                                 model_idx_2, layer_idx_2),
               root_dir=test_data_dir)
    # Return a mean summary of the principle angles
    # return np.mean(res_1[0:6]), np.mean(res_2[0:6])
    return np.mean(res_1), np.mean(res_2)


def main(args):
    # Load the activation folder (model to study)
    activation_dir = args.activation_dir
    model_idx = args.model_idx

    # # activation_dir sensitive param: (check before each job submission)
    data_partition_num = args.data_partition_num
    layer_idx_end = 7

    model_idx_1, model_idx_2 = model_idx, model_idx

    avg_angle_log = np.zeros([layer_idx_end, layer_idx_end])
    avg_angle_log_baselines = np.zeros_like(avg_angle_log)

    for layer_idx_1 in range(0, layer_idx_end):
        for layer_idx_2 in range(layer_idx_1, layer_idx_end):
            check_path_1 = os.path.join(activation_dir,
                                        'ds0_model%d_lay %d.npy' % (model_idx_1,
                                                                    layer_idx_1))
            check_path_2 = os.path.join(activation_dir,
                                        'ds0_model%d_lay %d.npy' % (model_idx_2,
                                                                    layer_idx_2)
                                        )
            if os.path.exists(check_path_1) and os.path.exists(check_path_2):
                # # Run analysis
                avg_test, avg_baseline = _subspace_compare(activation_dir, data_partition_num,
                                                           model_idx_1, model_idx_2,
                                                           layer_idx_1, layer_idx_2,
                                                           threshold=0.9)
                avg_angle_log[layer_idx_1, layer_idx_2] = avg_test
                avg_angle_log_baselines[layer_idx_1, layer_idx_2] = avg_baseline
            else:
                pass
    print(avg_angle_log)
    print(avg_angle_log_baselines)
    save_summary_dir = os.path.join(activation_dir, 'summary_mat')
    if not os.path.exists(save_summary_dir):
        os.mkdir(save_summary_dir)
    save_np_name = os.path.join(save_summary_dir, activation_dir[-8:])
    np.save(save_np_name + 'layer_res.npy', avg_angle_log)
    np.save(save_np_name + 'baseline.npy', avg_angle_log_baselines)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Specify 'data idx' and 'process batch size'.")

    parser.add_argument('--model_idx', dest='model_idx', type=int, action='store', default=0)
    parser.add_argument('--data_partition_num', dest='data_partition_num',
                        type=int, action='store', default=10)
    parser.add_argument('--activation_dir', dest='activation_dir', type=str, action='store',
                        default='model_activations/CBR_Tiny')

    args = parser.parse_args()
    main(args)

