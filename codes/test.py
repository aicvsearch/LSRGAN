import os
import sys
import time
import argparse
from collections import OrderedDict
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import PrintLogger
import numpy as np

def L1_loss(pre, gt):

    return np.sum(np.abs(pre-gt)) * 1.0 / pre.size / 255.0

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str,  default='./options/test/test_ESR.json', help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['L1'] = []
    test_results['Y_L1'] = []


    for data in test_loader:
        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True

        model.feed_data(data, need_HR=need_HR)
        img_path = data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.test()  # test
        visuals = model.get_current_visuals(need_HR=need_HR)

        sr_img = util.tensor2img(visuals['SR'])  # uint8

        if need_HR:  # load GT image and calculate psnr
            gt_img = util.tensor2img(visuals['HR'])

            crop_border = test_loader.dataset.opt['scale']
            cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]
            psnr = util.psnr(cropped_sr_img, cropped_gt_img)
            ssim = util.ssim(cropped_sr_img, cropped_gt_img, multichannel=True)

            l1_loss = L1_loss(cropped_sr_img, cropped_gt_img)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['L1'].append(l1_loss)
            if gt_img.shape[2] == 3:  # RGB image
                cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
                cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)
                psnr_y = util.psnr(cropped_sr_img_y, cropped_gt_img_y)
                ssim_y = util.ssim(cropped_sr_img_y, cropped_gt_img_y, multichannel=False)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                l1_y_loss = L1_loss(cropped_sr_img_y, cropped_gt_img_y)
                test_results['Y_L1'].append(l1_y_loss)

                print('{:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}; PSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}, L1: {:.4f}, Y_L1: {:.4f}'\
                    .format(img_name, psnr, ssim, psnr_y, ssim_y, l1_loss, l1_y_loss))
            else:
                print('{:20s} - PSNR: {:.4f} dB; SSIM: {:.4f}, L1: {:.4f}, Y_L1: {:.4f}'.format(img_name, psnr, ssim, l1_loss, l1_y_loss))
        else:
            print(img_name)

        suffix = opt['suffix']
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = os.path.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

    if need_HR:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_l1 = sum(test_results['L1']) / len(test_results['L1'])
        psnr_arr = np.array(test_results['psnr'])
        psnr_std = np.std(psnr_arr)
        ssim_arr = np.array(test_results['ssim'])
        ssim_std = np.std(ssim_arr)

        L1_arr = np.array(test_results['L1'])
        L1_std = np.std(L1_arr)
        L1_mean = np.mean((L1_arr))

        print('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.4f} dB; SSIM: {:.4f}, PSNR_std: {:.4f}, SSIM_std: {:.4f}, L1_mean: {:.4f}, L1_std: {:.4f}\n'\
                .format(test_set_name, ave_psnr, ave_ssim, psnr_std, ssim_std, L1_mean, L1_std))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

            psnr_y_arr = np.array(test_results['psnr_y'])
            psnr_y_std = np.std(psnr_y_arr)
            ssim_y_arr = np.array(test_results['ssim_y'])
            ssim_y_std = np.std(ssim_y_arr)
            Y_L1_arr = np.array(test_results['Y_L1'])
            Y_L1_std = np.std(Y_L1_arr)
            Y_L1_mean = np.mean(Y_L1_arr)

            print('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}; PSNR_Y_std: {:.4f} dB; SSIM_Y_std: {:.4f}, Y_L1_mean: {:.4f}, Y_L1_std: {:.4f}\n'\
                .format(ave_psnr_y, ave_ssim_y, psnr_y_std, ssim_y_std,Y_L1_mean, Y_L1_std))
