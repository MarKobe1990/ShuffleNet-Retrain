"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np

from misc_functions import (preprocess_image, get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from gradcam import GradCam
from guided_backprop import GuidedBackprop
from COME15KClassDataset import set_data_loader, OpenCVResize, COME15KDataSet
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


if __name__ == '__main__':
    dataset_dir = '../data/SOD-SemanticDataset-OriginalSize'
    dataset_attr_word = 'test_easy'

    # dataset_attr = class_txt_path_dic.get(dataset_attr_word)
    model_path = '../models/2023-10-09-01-47_max_epoch_100/'
    model_name = 'retrain_COME15K_checkpoint-best-avg-0.743-Medium.pth.tar'
    # init model
    model_and_weight_path = model_path + model_name
    shuffleNetV2PLUS = torch.load(model_and_weight_path)
    shuffleNetV2PLUS.to('cpu')
    pretrained_model = shuffleNetV2PLUS

    class_txt_path_dic = {
        "train": ['../data_class_txt/train_classes.txt', dataset_dir + '/train/'],
        "val_easy": ['../data_class_txt/val_easy_classes.txt', dataset_dir + '/test/COME15K-Easy/'],
        "val_hard": ['../data_class_txt/val_hard_classes.txt', dataset_dir + '/test/COME15K-Hard/'],
        "test_easy": ['../data_class_txt/test_easy_classes.txt', dataset_dir + '/test/COME15K-Easy/'],
        "test_hard": ['../data_class_txt/test_hard_classes.txt', dataset_dir + '/test/COME15K-Hard/'],
    }
    dataset_attr = class_txt_path_dic.get(dataset_attr_word)
    dataset = COME15KDataSet(txt_name=dataset_attr[0], data_path=dataset_attr[1], transform=None,
                             size=224, droup_out_class_label=None)
    datalist = dataset.images
    dataset_dic = {}
    for index, data_path in enumerate(datalist):
        dataset_dic.__setitem__(dataset_attr[1] + data_path, dataset.labels.__getitem__(index))
    pbar = tqdm(dataset_dic.items(), desc=dataset_attr_word, unit='images')

    for idx, (image_path, target) in enumerate(pbar):
        target_class = target
        # Read image
        original_image = Image.open(image_path).convert('RGB')
        # Process image
        prep_img = preprocess_image(original_image)
        file_name_to_export = image_path[image_path.rfind('/') + 1:image_path.rfind('.')]
        # Grad cam
        selected_module = 'features'
        stage = 'stage_four'
        stage_list = {"first_conv": [0, 16], "stage_one": [3, 48], "stage_two": [7, 128], "stage_three": [15, 256],
                      "stage_four": [19, 512]}
        target_stage_attr = stage_list.get(stage)
        gcv2 = GradCam(pretrained_model, selected_module, target_stage_attr)
        # Generate cam mask
        cam = gcv2.generate_cam(prep_img, target_class)
        print('Grad cam completed')

        # Guided backprop
        GBP = GuidedBackprop(pretrained_model)
        # Get gradients
        guided_grads = GBP.generate_gradients(prep_img, target_class)
        print('Guided backpropagation completed')
        # file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
        file_name_to_export = '1'
        # Guided Grad cam
        cam_gb = guided_grad_cam(cam, guided_grads)
        save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
        grayscale_cam_gb = convert_to_grayscale(cam_gb)
        save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
        print('Guided grad cam completed')

    # # Grad cam
    # gcv2 = GradCam(pretrained_model, target_layer=11)
    # # Generate cam mask
    # cam = gcv2.generate_cam(prep_img, target_class)
    # print('Grad cam completed')
    #
    # # Guided backprop
    # GBP = GuidedBackprop(pretrained_model)
    # # Get gradients
    # guided_grads = GBP.generate_gradients(prep_img, target_class)
    # print('Guided backpropagation completed')
    #
    # # Guided Grad cam
    # cam_gb = guided_grad_cam(cam, guided_grads)
    # save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    # grayscale_cam_gb = convert_to_grayscale(cam_gb)
    # save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    # print('Guided grad cam completed')
