import os
import torch
from torch.autograd import Variable
from torchvision.models import resnet50, vgg16
from torchvision import transforms
from matplotlib import pyplot as plt
import time
from torch.optim import Adam
import numpy as np
from scripts.utils import cca_core
import scipy
import PIL
from scripts.utils.model_cbr import CBR_Tiny, CBR_Small, CBR_LargeT, CBR_LargeW, CBR


normalise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def _fft2(array):
    return scipy.fft.fft2(array)


def _mag_phase_fft2(fft_array):
    return np.abs(fft_array), np.angle(fft_array)


def _ifft2(fft_mag, fft_phase):
    combined = np.multiply(fft_mag, np.exp(1j * fft_phase))
    return np.real(scipy.fft.ifft2(combined))


def _visualize_fft2(mag_array, phase_array,
                    save_dir,
                    clipping_threshold=250):
    """
    Visualize mag and phase plots and save the images to file.

    Args:
        mag_array:  array --- (num_image, channel=3, h, w)
        phase_array: array --- (num_image, channel=3, h, w)
        save_dir:   path to save the final plots
        clipping_threshold:   for better visualization of the mag plot

    Returns:

    """
    num_image, num_channel, _, _ = mag_array.shape
    for i in range(num_image):
        fig_1 = plt.figure()
        for c in range(num_channel):
            ax_1 = fig_1.add_subplot(2, num_channel, c+1)
            ax_1.imshow(np.clip(mag_array[i, c, :, :], -1, clipping_threshold), )
            ax_2 = fig_1.add_subplot(2, num_channel, num_channel+c+1)
            ax_2.imshow(phase_array[i, c, :, :])
        fig_name = save_dir + '_img%d.png' % i
        plt.savefig(fig_name, format='png')
        plt.close(fig_1)


def _visualize_fft_density(fft_mag, row_idx=None, col_idx=None, clipping_threshold=500):
    """
    This helper function plots two line of fft2_magnitude plot, which is helpful in finding the mag plot condition number.

    Args:
        fft_mag: the fft2 mag plt
        row_idx: the row plot
        col_idx: the col to plot
        clipping_threshold: for better vis quality in the 2d plot

    Returns:

    """
    assert row_idx is not None, 'Please Specify the Row to visualize'
    assert col_idx is not None, 'Please Specify the Col to visualize'
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(1, 3, 1)
    ax_1.imshow(np.clip(fft_mag, 0, clipping_threshold))
    ax_1.axvline(x=row_idx, color='red')
    ax_1.axhline(y=col_idx, color='green')
    ax_2 = fig_1.add_subplot(1, 3, 2)
    ax_2.plot(fft_mag[row_idx, :])
    ax_2.set_title('Red Line Mag')
    ax_3 = fig_1.add_subplot(1, 3, 3)
    ax_3.plot(fft_mag[:, col_idx])
    ax_3.set_title('Green Line Mag')
    plt.show()


def _find_energy_threshold(fft_mag, energy_percentage=0.9999):
    """
    A simple function to find the low-pass bandwidth of a fft2 image that preserves x% of the original information.

    Args:
        fft_mag:

    Returns:

    """
    h, _ = fft_mag.shape
    for margin in range(h):
        new_fft_mag = fft_mag.copy()
        new_fft_mag[margin:h-margin, :] = 0
        new_fft_mag[:, margin:h-margin] = 0
        if np.linalg.norm(new_fft_mag, ord=2) / np.linalg.norm(fft_mag, ord=2) > energy_percentage:
            return margin
    return None


def test_fft_phase(data_dir):
    """
        This is just a fun experiment to see how much sample points
        can be reduced using F-transform to reconstruct the original image.

    Args:
        data_dir: Contains image dataset in numpy array, (num_data, h, w, channels=3)

    Returns:
        Nothing. Just visualizing 2d-fft example with three plots.

    """
    test_img_1 = np.load(os.path.join(data_dir, 'data.npy'))[0, :, :, 0]
    test_img_2 = np.zeros(shape=test_img_1.shape)
    test_img_2[100:150, 100:150] = 0.5
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(1, 2, 1)
    ax_1.imshow(test_img_1)
    ax_2 = fig_1.add_subplot(1, 2, 2)
    ax_2.imshow(test_img_2)

    fft_1 = scipy.fft.fft2(test_img_1)
    fft_1_mag = np.abs(fft_1)
    fft_1_phase = np.angle(fft_1)
    fft_2 = scipy.fft.fft2(test_img_2)
    fft_2_mag = np.abs(fft_2)
    fft_2_phase = np.angle(fft_2)

    fig_2 = plt.figure()
    ax_1 = fig_2.add_subplot(2, 2, 1)
    ax_1.imshow(np.clip(fft_1_mag, 0, 50))
    ax_2 = fig_2.add_subplot(2, 2, 2)
    ax_2.imshow(fft_1_phase)
    ax_3 = fig_2.add_subplot(2, 2, 3)
    ax_3.imshow(np.clip(fft_2_mag, 0, 50))
    ax_3 = fig_2.add_subplot(2, 2, 4)
    ax_3.imshow(fft_2_phase)

    combined_1 = np.multiply(fft_1_mag, np.exp(1j * fft_1_phase))
    combined_2 = np.multiply(fft_2_mag, np.exp(1j * fft_1_phase))
    combined_1 = np.real(scipy.fft.ifft2(combined_1))
    combined_2 = np.real(scipy.fft.ifft2(combined_2))
    fig_3 = plt.figure()
    ax_1 = fig_3.add_subplot(1, 2, 1)
    ax_1.imshow(combined_1)
    ax_2 = fig_3.add_subplot(1, 2, 2)
    ax_2.imshow(combined_2)

    h, w = fft_1.shape
    margin = 50
    fft_3_mag, fft_3_phase = fft_1_mag.copy(), fft_1_phase.copy()
    fft_3_mag[margin:h-margin, :] = 0
    fft_3_mag[:, margin:h - margin] = 0
    fft_3_phase[margin:h-margin, :] = 0
    fft_3_phase[:, margin:h - margin] = 0
    combined_3 = np.multiply(fft_3_mag, np.exp(1j * fft_3_phase))
    combined_3 = np.real(scipy.fft.ifft2(combined_3))

    fig_4 = plt.figure()
    ax_1 = fig_4.add_subplot(1, 3, 1)
    ax_1.imshow(np.clip(fft_3_mag, 0, 50))
    ax_2 = fig_4.add_subplot(1, 3, 2)
    ax_2.imshow(fft_3_phase)
    ax_3 = fig_4.add_subplot(1, 3, 3)
    ax_3.imshow(combined_3)
    ax_3.set_title('Low Passed Reconstruction')

    margin_2 = _find_energy_threshold(fft_1_mag)
    fft_4_mag, fft_4_phase = fft_1_mag.copy(), fft_1_phase.copy()
    fft_4_mag[margin_2:h - margin_2, :] = 0
    fft_4_mag[:, margin_2:h - margin_2] = 0
    fft_4_phase[margin_2:h - margin_2, :] = 0
    fft_4_phase[:, margin_2:h - margin_2] = 0
    combined_4 = _ifft2(fft_4_mag, fft_4_phase)
    print('99% energy Margin: ', margin_2)
    fig_5 = plt.figure()
    ax_1 = fig_5.add_subplot(1, 2, 1)
    ax_2 = fig_5.add_subplot(1, 2, 2)
    ax_1.imshow(test_img_1)
    ax_1.set_title('Original')
    ax_2.imshow(combined_3)
    ax_2.set_title('Low-Pass Reconstructed (Margin=50)')

    plt.show()
    plt.close(fig_1)
    plt.close(fig_2)
    plt.close(fig_3)
    plt.close(fig_4)


def init_image(size=(400, 400, 3)):
    # Just to make sure each generated random number are different
    np.random.seed(int(time.time() % 19))
    img = PIL.Image.fromarray(np.uint8(np.random.uniform(150, 180, size)))
    img_tensor = normalise(img).unsqueeze(0)
    return img_tensor


def to_variable(image, device, requires_grad=False):
    image = Variable(image.to(device, dtype=torch.float32), requires_grad=requires_grad)
    return image


def _filter_step(model, img, device, step_size=5, filter_idx=None):
    mean_ = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std_ = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    model.zero_grad()

    img_var = to_variable(img, device=device, requires_grad=True)
    optimizer = Adam([img_var], lr=step_size)

    x = model(img_var)
    if filter_idx is None:
        output = x[0, :, :, :]
    else:
        output = x[0, filter_idx, :, :]
    loss = -(output.norm() - torch.mean(img_var ** 2))  # torch.mean(output)
    # loss = -(torch.sum(output ** 2) - torch.sum(img_var ** 2))  # torch.mean(output)
    loss.backward()
    optimizer.step()

    result = img_var.data.cpu().numpy()
    result[0, :, :, :] = np.clip(result[0, :, :, :], -mean_ / std_, (1 - mean_) / std_)
    return torch.Tensor(result), loss.detach().data.cpu().numpy()


def _visualize_filter(model, base_img, iter_n, device, step_size, filter_idx):
    loss_log = []
    for i in range(iter_n):
        base_img, loss = _filter_step(model, base_img, device, step_size, filter_idx=filter_idx)
        loss_log.append(loss)
    return base_img, loss_log


def _generate_activation(test_module, iter_n, step_size, device,
                         input_size=(50,50,3), filter_idx=None):
    """
    Generate one activation max example given the requirements.

    Args:
        test_module:  A (subset of the) testing NN network.
        iter_n:      # of iterations to optimize the input image
        step_size:   step to alter the input image at each iteration
        device: e.g., torch.device('gpu')
        input_size:   input image size, at least larger than the perceptive field
        filter_idx:   None --- using average layer activation; otherwise is the channel idx to activate.

    Returns: The activation max result in numpy array with input_size; a list of loss values

    """
    input_tensor = init_image(size=input_size)
    activation, loss_log = _visualize_filter(model=test_module,
                                             base_img=input_tensor,
                                             iter_n=iter_n,
                                             device=device,
                                             step_size=step_size,
                                             filter_idx=filter_idx)
    return activation.detach().cpu().data.numpy()[0, :, :, :], loss_log


def _wrap_visualizations(activation_map,
                         loss_log,
                         fig_log_dir,
                         module_name,
                         module_idx,
                         filter_name,
                         trial_idx):
    """
    Save to file: 1) the generated activation map  2) the optimization loss (to check if the activation map converges)
    Args:
        activation_map:  e.g., the result from the function _generate_activation
        loss_log:        e.g., the result from the function _generate_activation
        fig_log_dir:
        module_name:
        module_idx:
        filter_name:
        trial_idx:      the idx number of the generated example to track result
    """
    _plot_activation_map(activation_map=activation_map,
                         fig_log_dir=fig_log_dir,
                         module_name=module_name,
                         module_idx=module_idx,
                         filter_name=filter_name,
                         trial_number=trial_idx)
    _plot_loss_log(loss_log=loss_log,
                   fig_log_dir=fig_log_dir,
                   module_name=module_name,
                   module_idx=module_idx,
                   filter_name=filter_name,
                   trial_number=trial_idx)


def _plot_activation_map(activation_map, fig_log_dir,
                         module_name,
                         module_idx,
                         filter_name,
                         trial_number):
    # ===== Visualize Generated Activations
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(1, 1, 1)
    ax_1.imshow(np.transpose(activation_map, [1, 2, 0]))
    save_fig_dir = os.path.join(fig_log_dir, '%s%d_Filter%s_Trial%d.png' % (module_name,
                                                                            module_idx,
                                                                            filter_name,
                                                                            trial_number))
    plt.savefig(save_fig_dir, format='png')
    plt.close(fig_1)


def _plot_loss_log(loss_log,
                   fig_log_dir,
                   module_name,
                   module_idx,
                   filter_name,
                   trial_number
                   ):
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(1, 1, 1)
    ax_1.plot(loss_log)
    ax_1.set_title('Activation Max Log')
    save_fig_dir = os.path.join(fig_log_dir, '%s%d_Filter%s_Trial%d_loss.png' % (module_name,
                                                                                 module_idx,
                                                                                 filter_name,
                                                                                 trial_number))
    plt.savefig(save_fig_dir, format='png')
    plt.close(fig_1)


def _fft_analysis(root_dir,
                  use_model,
                  avg_activation=False):
    activation_map_path = os.path.join(root_dir,
                                       use_model)
    if avg_activation:
        activation_map_path = os.path.join(activation_map_path,
                                           'avg')
        data_file_lst = [f for f in os.listdir(activation_map_path) if (os.path.isfile(os.path.join(activation_map_path,
                                                                                                    f)) and (
                                                                                    '.npy' in f))]
        _save_fft_plots(root_dir=activation_map_path, file_lst=data_file_lst)
    else:
        activation_map_path = os.path.join(activation_map_path,
                                           'per-filter-activation')
        module_dir_lst = [f_dir for f_dir in os.listdir(activation_map_path)]
        for module_dir in module_dir_lst:
            module_path = os.path.join(activation_map_path,
                                       module_dir)
            data_file_lst = [f for f in os.listdir(module_path) if
                             (os.path.isfile(os.path.join(module_path,
                                                          f)) and (
                                      '.npy' in f))]
            _save_fft_plots(root_dir=module_path, file_lst=data_file_lst)
    print('FFT plots saved to file.')


def _save_fft_plots(root_dir, file_lst):
    save_plots_dir = os.path.join(root_dir,
                                  'fft2_vis')
    if not os.path.exists(save_plots_dir):
        os.mkdir(save_plots_dir)
    for file in file_lst:
        data_file = os.path.join(root_dir,
                                 file)
        activation = np.load(data_file)
        num_data, channel, h, w = activation.shape
        mag_res, phase_res = [], []
        for i in range(num_data):
            mag_channel_res, phase_channel_res = [], []
            for c in range(channel):
                test_feature = activation[i, c, :, :]
                fft_feature = _fft2(test_feature)
                mag, phase = _mag_phase_fft2(fft_feature)
                mag_channel_res.append(mag)
                phase_channel_res.append(phase)
            mag_channel_res = np.stack(mag_channel_res, axis=0)
            phase_channel_res = np.stack(phase_channel_res, axis=0)
            mag_res.append(mag_channel_res)
            phase_res.append(phase_channel_res)
        mag_res = np.stack(mag_res, axis=0)
        phase_res = np.stack(phase_res, axis=0)
        plot_name = os.path.join(save_plots_dir, file[:-4])
        np.save(plot_name+'_mag.npy', mag_res)
        np.save(plot_name + '_phase.npy', phase_res)
        _visualize_fft2(mag_res, phase_res,
                        clipping_threshold=500,
                        save_dir=plot_name)


def _act_max(save_activation_dir,
             model,
             model_type,
             block_name=None,
             avg_activation=True,
             num_trial=1000,
             input_size=(224, 224, 3),
             iter_n=500,
             step_size=1e-1):
    # These two params are just for demos, need to refine if this script turns into formal analysis
    filter_start, filter_end = 10, 20

    assert block_name is not None, 'NN block seperation should not be None.'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    if model_type == 'resnet':
        module_lst = list(model.children())
        module_name = 'BottleNeck'
    elif model_type == 'vgg':
        module_lst = list(model.features.children())
        module_name = 'CovRelu'
    elif model_type == 'CBR':
        module_lst = list(model.children())
        module_name = 'CBR'
    else:
        print('Model type unsupported yet. Check configuration.')
        return None
    if avg_activation:
        filter_name = 'avg'
        log_root_dir = os.path.join(save_activation_dir, filter_name)
        if not os.path.exists(log_root_dir):
            os.mkdir(log_root_dir)
        for idx in range(len(module_lst)):
            if type(module_lst[idx]) == block_name:
                module_log_dir = log_root_dir
                res_to_save = []
                for trial_idx in range(num_trial):
                    test_module = torch.nn.Sequential(*module_lst[0:idx + 1])
                    activation_map, loss_log = _generate_activation(test_module,
                                                                    iter_n,
                                                                    step_size,
                                                                    device,
                                                                    input_size=input_size,
                                                                    filter_idx=3)
                    res_to_save.append(activation_map)
                    # # ===== Visualize Generated Activations
                    _wrap_visualizations(activation_map=activation_map,
                                         loss_log=loss_log,
                                         fig_log_dir=module_log_dir,
                                         module_name=module_name,
                                         module_idx=idx,
                                         filter_name=filter_name,
                                         trial_idx=trial_idx)
                save_dir = os.path.join(module_log_dir,
                                        '%s_%d.npy' % (module_name, idx))
                save_array = np.stack(res_to_save, axis=0)
                print(save_array.shape)
                np.save(save_dir, save_array)
    else:
        log_root_dir = os.path.join(save_activation_dir, 'per-filter-activation')
        if not os.path.exists(log_root_dir):
            os.mkdir(log_root_dir)
        for idx in range(len(module_lst)):
            if type(module_lst[idx]) == block_name:
                module_log_dir = os.path.join(log_root_dir, 'Module%d' % idx)
                if not os.path.exists(module_log_dir):
                    os.mkdir(module_log_dir)
                for filter_idx in range(filter_start, filter_end, 1):
                    filter_name = str(filter_idx)
                    res_to_save = []
                    for trial_idx in range(num_trial):
                        test_module = torch.nn.Sequential(*module_lst[0:idx + 1])
                        activation_map, loss_log = _generate_activation(test_module,
                                                                        iter_n,
                                                                        step_size,
                                                                        device,
                                                                        input_size=input_size,
                                                                        filter_idx=filter_idx)
                        res_to_save.append(activation_map)
                        # # ===== Visualize Generated Activations
                        _wrap_visualizations(activation_map=activation_map,
                                             loss_log=loss_log,
                                             fig_log_dir=module_log_dir,
                                             module_name=module_name,
                                             module_idx=idx,
                                             filter_name=filter_name,
                                             trial_idx=trial_idx)
                    save_dir = os.path.join(module_log_dir,
                                            '%s_%d_filter_%s.npy' % (module_name, idx, filter_name))
                    save_array = np.stack(res_to_save, axis=0)
                    print(save_array.shape)
                    np.save(save_dir, save_array)


if __name__ == "__main__":
    # Change use_model to specify testing model
    use_model = 'CBR_Tiny'
    num_trial = 2  # number of activation to generate on each layer/block
    # avg_activation = True to use avg_activation; False to use per filter activation
    # But if you decide to use False, go check filter_start and filter_end parameter inside
    # function "_act_max()"
    avg_activation = False

    #  DATA_DIR points at where Messidor1 dataset (npy) locates
    DATA_DIR = os.path.abspath('dataset/Messidor1')
    # To see the toy visualization of fft2d and the importance of phase plot run the following line
    # test_fft_phase(DATA_DIR)

    # ==== Generating Activation Max from different depth of layers ==== #
    # Activation Max Feature Root Dir
    activation_feature_dir = os.path.abspath('act_max_features')
    if not os.path.exists(activation_feature_dir):
        os.mkdir(activation_feature_dir)

    # Specified configurations and log directories based on parameter 'use_model'
    feature_dir = os.path.join(activation_feature_dir, use_model)
    # if not os.path.exists(feature_dir):
    #     os.mkdir(feature_dir)
    # if use_model == 'vgg':
    #     model = vgg16(pretrained=True)
    #     model_type = 'vgg'
    #     split_block = torch.nn.ReLU
    # elif use_model == 'resnet':
    #     # model = resnet50(pretrained=False, num_classes=2)
    #     # MODEL_DIR = os.path.abspath('saved_models/resnet50.pt')
    #     # model.load_state_dict(torch.load(MODEL_DIR))
    #     model = resnet50(pretrained=True)
    #     model_type = 'resnet'
    #     split_block = torch.nn.Sequential
    # elif use_model == 'CBR_Tiny':
    #     model = CBR_Tiny()
    #     MODEL_DIR = os.path.abspath('saved_models/CBR_Tiny.pt')
    #     model.load_state_dict(torch.load(MODEL_DIR))
    #     model_type = 'CBR'
    #     split_block = CBR
    # elif use_model == 'CBR_Small':
    #     model = CBR_Small()
    #     MODEL_DIR = os.path.abspath('saved_models/CBR_Small.pt')
    #     model.load_state_dict(torch.load(MODEL_DIR))
    #     model_type = 'CBR'
    #     split_block = CBR
    # else:
    #     print('Not supporting model type yet.')

    # # OBS: if avg_activation is set as "False", check filter_start and filter_end
    # # inside function _act_max, right now they are not automatically set to adapt to model types
    # _act_max(save_activation_dir=feature_dir,
    #          model=model,
    #          model_type=model_type,
    #          block_name=split_block,
    #          avg_activation=avg_activation,
    #          num_trial=num_trial,
    #          input_size=(224, 224, 3),
    #          iter_n=1000,
    #          step_size=1e-1)

    # # === Fourier Analysis of the activation maps === # #
    # _fft_analysis(root_dir=activation_feature_dir,
    #               use_model=use_model,
    #               avg_activation=avg_activation)

    # # === Analysis Energy (low-passed) Percentage of the activation maps ===

    print('That\'s it! No more sleep and get up now!')

