"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.utils as utils
import torchvision.transforms as transforms
import torch
from torch.optim import Adam

from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, stage, model, selected_module, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.stage = stage
        self.selected_module = selected_module
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('./generated'):
            os.makedirs('./generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model.first_conv):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if self.selected_module == 'first_conv' and index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            for index, layer in enumerate(self.model.features):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if self.selected_module == 'features' and index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 30 == 0:
                im_path = './generated/' + self.stage + '_layer_vis_m' + str(self.selected_module) + '_l' + str(self.selected_layer) + \
                          '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model.first_conv):
                # Forward pass layer by layer
                x = layer(x)
                if self.selected_module == 'first_conv' and index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            for index, layer in enumerate(self.model.features):
                # Forward pass layer by layer
                x = layer(x)
                if self.selected_module == 'features' and index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 30 == 0:
                im_path = './generated/' + self.stage + '_layer_vis_l' + str(self.selected_layer) + \
                          '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)
                return self.created_image


if __name__ == '__main__':
    model_path = '../models/2023-10-09-01-47_max_epoch_100/'
    model_name = 'retrain_COME15K_checkpoint-best-avg-0.743-Medium.pth.tar'
    # init model
    model_and_weight_path = model_path + model_name
    shuffleNetV2PLUS = torch.load(model_and_weight_path)
    shuffleNetV2PLUS.to('cpu')
    pretrained_model = shuffleNetV2PLUS
    selected_module = 'features'
    stage = 'stage_four'
    stage_list = {"first_conv": [0, 16], "stage_one": [3, 48], "stage_two": [7, 128], "stage_three": [15, 256],
                  "stage_four": [19, 512]}
    cnn_layer = stage_list[stage][0]
    # filter_pos = 5
    # Fully connected layer is not needed
    # all filter
    img_np_list = []
    img_tensor_list = []
    for filter_pos in range(0, stage_list[stage][1]):
        layer_vis = CNNLayerVisualization(stage, pretrained_model, selected_module, cnn_layer, filter_pos)
        # Layer visualization with pytorch hooks
        # layer_vis.visualise_layer_with_hooks()

        # Layer visualization without pytorch hooks
        created_image = layer_vis.visualise_layer_without_hooks()
        img_np_list.append(created_image)
    for img_np in img_np_list:
        img_tensor = transforms.ToTensor()(img_np).unsqueeze(dim=0)
        img_tensor_list.append(img_tensor)
    image_grid = utils.make_grid(torch.cat(img_tensor_list, dim=0), int(stage_list[stage][1] / 4)).clone().detach().to(torch.device('cpu'))
    im_path = './generated/' + stage + '_layer_vis_m' + str(selected_module) + '_l' + str(cnn_layer) + \
              '_ALL_filter' + '_iter' + str(30) + '.jpg'
    utils.save_image(image_grid, im_path)
    # cnn_layer = 17
    # filter_pos = 5
    # # Fully connected layer is not needed
    # pretrained_model = models.vgg16(pretrained=True).features
    # layer_vis = CNNLayerVisualization(pretrained_model, selected_module,cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    # layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
