# SimpleNN: Neural Networks from Scratch
## Introduction
SimpleNN is a demonstration project aimed at showcasing an implementation of the essential components of neural networks in C++. Developed as a portfolio piece, it serves as a testament to understanding the inner workings of neural architectures and the mathematics underpinning them. By sidestepping the abstractions offered by major deep learning frameworks, this repo is a transparent view into the core operations of neural networks.

## Features
**Basic Neural Layers**: Experience the simplicity with fundamental layers like Dense, Dropout and Flatten.

**Activation Functions**: Delve into popular activation functions, including ReLU and Sigmoid, that drive the non-linearity in neural networks.

**Forward & Backward Propagation**: A hands-on approach to the crucial processes of training a neural network, illustrating both forward and backward passes.

**Eigen Integration**: Leveraging the Eigen library for efficient matrix operations, offering a glimpse into the high-performance computation required in deep learning.

## Purpose
The primary goal of SimpleNN is educational, provide understanding of neural networks without the complexities of extensive frameworks.

## ToDo
- Add different metrics and history object with ability of correpsonding plot in Matplot
- Add optimizers: Momentum, RMSProp, Adam, AdaGrad
- Advanced dataset split (like sklearn *test_train_split* function).
- Image input. For now, it is difficult to implement directly because Eigen::MatrixXf (default container for now) requires too much space when loading an image. So, probably we need to implement our own tensor-like container or/and use c++ templates.
- In addition to image input. Have something like Keras ImageDataGenerator would be nice.
- Refinements of code. Appropriate implementation for test_cases.
- Regularization: L1, L2, ElasticNet. Need to come up with some agile implementation of these methods (don't want to hard-code them into each cost object)
- Normalization mechanism: dataset, batch, inner-layer
- Question about resourse optimization in Dense
- Separate mini-batch shuffling mechanism and Sequential class.