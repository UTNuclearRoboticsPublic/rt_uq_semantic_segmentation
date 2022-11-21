"""
    Maintainer: Selma Wanna  (slwanna@utexas.edu)

    This file creates the FastSCNN model architecture.
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "stochman"))
import torch
import torch.nn as nn
from stochman import nnj


class FastSCNN(nn.Module):
    """
    Constructs the FastSCNN model
    """
    def __init__(self, last_layer: bool=False, num_classes:int=2):
        super().__init__()

        self.last_layer = last_layer
        self.num_classes = num_classes

        # Model layer sizing
        self.dw_channels1 = 32
        self.dw_channels2 = 64
        self.out_channels = 128

        """
        FastSCNN full architecture
        """
        self.deterministic = nnj.Sequential(
            # 1. Learning to Downsample:
            nnj.Conv2d(3, self.dw_channels1, 1, bias=False),
            nnj.Tanh(),
            nnj.Conv2d(self.dw_channels1, self.dw_channels1, 1, bias=False),
            nnj.Tanh(),
            nnj.Conv2d(self.dw_channels1, self.dw_channels2, 1, bias=False),
            nnj.ReLU(),
            nnj.Conv2d(self.dw_channels2, self.dw_channels2, 1, bias=False),
            nnj.Tanh(),
            nnj.Conv2d(self.dw_channels2, self.out_channels, 1, bias=False),
            nnj.Tanh(),
            nnj.Flatten(),
            nnj.SkipConnection(
                nnj.Reshape(128,256,256),
                # 2. Global Feature Extractor:
                #     a) linear bottle neck
                nnj.Conv2d(128,64*6,1, bias=False),
                nnj.Tanh(),
                nnj.Conv2d(64*6,64*6,1, bias=False),
                nnj.Tanh(),
                nnj.Conv2d(64*6,64,1,bias=False),
                #     b) linear bottle neck
                nnj.Conv2d(64,96*6,1, bias=False),
                nnj.Tanh(),
                nnj.Conv2d(96*6,96*6,1, bias=False),
                nnj.Tanh(),
                nnj.Conv2d(96*6,96,1,bias=False),
                #     c) linear bottle neck
                nnj.Conv2d(96,128*6,1, bias=False),
                nnj.Tanh(),
                nnj.Conv2d(128*6,128*6,1, bias=False),
                nnj.Tanh(),
                nnj.Conv2d(128*6,128*6,1,bias=False),
        #     #     # #     d) pyramid pooling layer -- Can't work w/ Sequential, needs 4 features
        #     #     # # nnj.MaxPool2d(1),
        #     #     # # nnj.Conv2d(128,128,1, bias=False), 
        #     #     # # nnj.Upsample(scale_factor=2), # feature 1
        #     #     # # nnj.MaxPool2d(2),
        #     #     # # nnj.Conv2d(128,128,1, bias=False), 
        #     #     # # nnj.Upsample(scale_factor=2),  # feature 2
        #     #     # # nnj.MaxPool2d(3),
        #     #     # # nnj.Conv2d(128,128,1, bias=False), 
        #     #     # # nnj.Upsample(scale_factor=2), # feature 3
                nnj.MaxPool2d(2),
                # nnj.Conv2d(768,128,1, bias=False), 
                # nnj.Upsample(scale_factor=2),      # feature 4
                nnj.Flatten(),
                add_hooks=True,
            ),
            # Skipping Feature fusion
            # 3. Feature Fusion:
            nnj.Reshape(320,256,256),
            # # 4. Classifier
            nnj.Conv2d(320,256,1),
            nnj.Tanh(),
            nnj.Conv2d(256,256,1),
            nnj.Tanh(),
            nnj.Conv2d(256,256,1),
            nnj.Tanh(),

            nnj.Conv2d(256,256,1),
            nnj.Tanh(),
            nnj.Conv2d(256,256,1),
            nnj.Tanh(),

            nnj.Conv2d(256,self.num_classes,1)

        )


    def forward(self, x):
        """
        Executes forward pass
        """
        return self.deterministic(x)



if __name__ == '__main__':
    img = torch.randn(2, 3, 256, 256)
    model = FastSCNN()
    print(model(img).shape)