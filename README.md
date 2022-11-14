# Digit Recognition
This project was done as part of the Machine Learning with Python-From Linear Models to Deep Learning course [MITx MicroMasters in Statistics and Data Science]( https://micromasters.mit.edu/ds/ )

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project.  I got a chance to experiment with the task of classifying these images into the correct digit using some of the methods I have learned in the course.

|Data|Sample| Classification |
|-|-|-|
|MNIST dataset|<img src="https://user-images.githubusercontent.com/77168758/201578210-5ad15f65-d922-412d-aaad-eb1b2b710d86.png" width="50">| 6 |
|MNIST Two digit dataset|<img src="https://user-images.githubusercontent.com/77168758/201577472-14da0dbf-a5e6-4e04-a1b0-685da4472b74.png" width="50">| (4, 3) |

## Functions implemeted by me
1. [part1/linear_regression.py](https://github.com/tkayalvizhi/digit_recognizer/blob/016e19227fbbfd75beb1612239e20705897f40f5/part1/linear_regression.py) (linear regression)
2. [part1/svm.py](https://github.com/tkayalvizhi/digit_recognizer/blob/016e19227fbbfd75beb1612239e20705897f40f5/part1/svm.py) (support vector machine)
3. [part1/softmax.py](https://github.com/tkayalvizhi/digit_recognizer/blob/016e19227fbbfd75beb1612239e20705897f40f5/part1/softmax.py) (multinomial regression)
4. [part1/features.py/project_onto_PC](https://github.com/tkayalvizhi/digit_recognizer/blob/016e19227fbbfd75beb1612239e20705897f40f5/part1/features.py#L5) (principal component analysis (PCA) dimensionality reduction)
5. [part1/kernel.py](https://github.com/tkayalvizhi/digit_recognizer/blob/016e19227fbbfd75beb1612239e20705897f40f5/part1/kernel.py) (polynomial and Gaussian RBF kernels)
6. [part1/main.py](https://github.com/tkayalvizhi/digit_recognizer/blob/016e19227fbbfd75beb1612239e20705897f40f5/part1/main.py)
7. [part2-nn/neural_nets.py](https://github.com/tkayalvizhi/digit_recognizer/blob/d31a4edce8bee0f2cf2e30880e7c4e759c49ea97/part2-nn/neural_nets.py) implemented neural net from scratch
8. [part2-mnist/nnet_fc.py](https://github.com/tkayalvizhi/digit_recognizer/blob/d31a4edce8bee0f2cf2e30880e7c4e759c49ea97/part2-mnist/nnet_fc.py) used PyTorch to classify MNIST digits
9. [part2-mnist/nnet_cnn.py](https://github.com/tkayalvizhi/digit_recognizer/blob/d31a4edce8bee0f2cf2e30880e7c4e759c49ea97/part2-mnist/nnet_cnn.py) used convolutional layers to boost performance
10. [part2-twodigit/mlp.py](https://github.com/tkayalvizhi/digit_recognizer/blob/d31a4edce8bee0f2cf2e30880e7c4e759c49ea97/part2-twodigit/mlp.py) and [part2-twodigit/conv.py](https://github.com/tkayalvizhi/digit_recognizer/blob/d31a4edce8bee0f2cf2e30880e7c4e759c49ea97/part2-twodigit/conv.py) which are for a new, more difficult version of the MNIST dataset
