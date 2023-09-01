"""
This is a Python script that defines a PyTorch neural network called "parsingNet" for lane detection. The network takes an image as input and outputs a tensor of predicted lane locations.

The network is based on the ResNet architecture and consists of several convolutional, batch normalization, and ReLU layers. There are also some auxiliary heads that are used for multi-task learning, in addition to the main lane detection head.

The input image is first passed through the ResNet backbone, which outputs feature maps at multiple resolutions. These feature maps are then processed by the auxiliary heads, which produce intermediate lane predictions. The feature maps are also passed through a global pooling layer, which reduces their spatial dimensions and produces a feature vector that is fed to the main lane detection head.

The main lane detection head consists of two fully connected layers that output a tensor of size (num_gridding, num_cls_per_lane, num_of_lanes), where num_gridding is the number of rows in the output tensor, num_cls_per_lane is the number of column anchors per lane, and num_of_lanes is the number of lanes.

The output tensor is then reshaped to produce a tensor of size (batch_size, num_of_lanes, num_gridding, num_cls_per_lane), which represents the predicted locations of the lanes in the image. If the network is trained with the auxiliary heads, it also outputs the intermediate lane predictions.
"""



import torch
from ultrafastLaneDetector.backbone import resnet
import numpy as np

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        # Define convolutional layer
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        # Define batch normalization layer
        self.bn = torch.nn.BatchNorm2d(out_channels)
        # Define ReLU activation layer
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        # Pass input through convolutional layer
        x = self.conv(x)
        # Pass result through batch normalization layer
        x = self.bn(x)
        # Pass result through ReLU activation layer
        x = self.relu(x)
        # Return output
        return x

class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        # Define the size of the input images
        self.size = size
        self.w = size[0]
        self.h = size[1]

        # Define the number of dimensions for classification output
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors

        # Specify whether to use an auxiliary segmentation task
        self.use_aux = use_aux

        # Calculate the total number of dimensions
        self.total_dim = np.prod(cls_dim)

        # Define the backbone network
        self.model = resnet(backbone, pretrained=pretrained)

        # If using an auxiliary segmentation task, define the corresponding layers
        if self.use_aux:
            # Define the auxiliary header 2
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            # Define the auxiliary header 3
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )
            # Define the auxiliary header 4
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128,128,3,padding=1),
            )

            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        # 1/32,2048 channel
        # 288,800 -> 9,40,2048
        # (w+1) * sample_rows * 4
        # 37 * 10 * 4
        #initialize_weights(self.cls)

    def forward(self, x):
        # Apply the model to the input
        x2,x3,fea = self.model(x)

        # If using auxiliary segmentation headers, pass the feature maps through them and combine their outputs
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        # Pass the final feature map through max pooling and flatten it
        fea = self.pool(fea).view(-1, 1800)

        # Pass the flattened feature map through the classification module
        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        # Return the classification output and the auxiliary segmentation output if using auxiliary headers
        if self.use_aux:
            return group_cls, aux_seg

        return group_cls



def initialize_weights(*models):
    # Loop over the input models
    for model in models:
        # Call the real_init_weights function for each model
        real_init_weights(model)

def real_init_weights(m):
    # Check if the input module is a list
    if isinstance(m, list):
        # If it is a list, loop over the modules inside the list
        for mini_m in m:
            # Call the real_init_weights function for each module in the list
            real_init_weights(mini_m)
    else:
        # If it is not a list, check the module type
        if isinstance(m, torch.nn.Conv2d):    
            # If it is a convolutional layer, initialize the weights using kaiming_normal method with relu nonlinearity
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # If the layer has a bias term, set it to 0
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            # If it is a linear layer, initialize the weights using normal distribution with mean 0 and standard deviation 0.01
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            # If it is a batch normalization layer, set the weights to 1 and bias to 0
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            # If it is a module, loop over its children modules and call the real_init_weights function recursively
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            # If it is not a recognized module, print a message indicating so
            print('unknown module', m)
