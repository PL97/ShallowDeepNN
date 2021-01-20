import os
import numpy as np
import argparse


def main(args):
    act_data_folder = args.activation_dir
    print('Checking file folder: ', act_data_folder)
    act_data_list = [f for f in os.listdir(act_data_folder) if
                     (os.path.isfile(os.path.join(act_data_folder, f)) and ('.npy' in f))]
    for file in act_data_list:
        print('Checking file: ', file)
        file_path = os.path.join(act_data_folder, file)
        data = np.load(file_path)
        print('Stored Data Dimension: ', data.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check the saved nparray dimensions in a folder.")
    parser.add_argument('--activation_dir', dest='activation_dir', type=str, action='store',
                        default='model_activations/CBR_Tiny')
    main(args=parser.parse_args())
    print('Done!')

