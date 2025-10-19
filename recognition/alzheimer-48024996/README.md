# Alzheimer's Disease Classification using GFNet

## Project Overview
This project implements a binary classification model based on GFNet to distinguish between normal and Alzheimer's disease (AD) brain scans from the ADNI dataset. The goal is to achieve a minimum accuracy of $0.8$ on the test set.

To implement this, PyTorch will be used to develop and train the GFNet architecture.

## GFNet (Global Filter Network)
GFNet is a new neural network architecture that replaces the normal local convolution operation with global filtering in the frequency domain. Instead of learning spatial kernels (like in a normal CNN), GFNet performs a 2D Fourier transform on input feature map, applies learnable filters in the frequency domain to capture long range dependencies efficiently, then transforms the result back into the spatial domain using the inverse Fourier transformation.

Basically, GFNet treats the image as a global signal and learns how to manipulate its frequency components directly, rather than relying on local spatial features extracted by small convolutional kernels.

This allows GFNet to model global relationships across the entire image in a single operation. As a result, it achieves a balance between the efficiency of CNNs and the long range dependency modelling of transformers.

In this project, we will be using the GFNetPyramid model (which uses a hierarchical architecture to learn smaller details of the image in the early layers and more abstract, global features in the later layers) instead of the standard GFNet model.