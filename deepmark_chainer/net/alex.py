from chainer import link
from chainer.functions.activation import relu
from chainer.functions.pooling import max_pooling_2d
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear


class Alex(link.Chain):
    """AlexNet without normalization and dropout.

    Reference: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
    Caffe prototxt: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt

    ``Alex`` assumes its inputs be 4-dimentional tensors of the shape (B, 3, 224, 224).

    """

    def __init__(self, use_cudnn=True):
        super(Alex, self).__init__(
            conv1a=convolution_2d.Convolution2D(3, 48, 11, stride=4, pad=2, use_cudnn=use_cudnn),
            conv1b=convolution_2d.Convolution2D(3, 48, 11, stride=4, pad=2, use_cudnn=use_cudnn),
            conv2a=convolution_2d.Convolution2D(48, 128, 5, pad=2, use_cudnn=use_cudnn),
            conv2b=convolution_2d.Convolution2D(48, 128, 5, pad=2, use_cudnn=use_cudnn),
            conv3a=convolution_2d.Convolution2D(128, 192, 3, pad=1, use_cudnn=use_cudnn),
            conv3b=convolution_2d.Convolution2D(128, 192, 3, pad=1, use_cudnn=use_cudnn),
            conv3c=convolution_2d.Convolution2D(128, 192, 3, pad=1, use_cudnn=use_cudnn),
            conv3d=convolution_2d.Convolution2D(128, 192, 3, pad=1, use_cudnn=use_cudnn),
            conv4a=convolution_2d.Convolution2D(192, 192, 3, pad=1, use_cudnn=use_cudnn),
            conv4b=convolution_2d.Convolution2D(192, 192, 3, pad=1, use_cudnn=use_cudnn),
            conv5a=convolution_2d.Convolution2D(192, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv5b=convolution_2d.Convolution2D(192, 128, 3, pad=1, use_cudnn=use_cudnn),
            fc6a=linear.Linear(4608, 4096),
            fc6b=linear.Linear(4608, 4096),
            fc7=linear.Linear(4096, 4096),
            fc8=linear.Linear(4096, 1000))

    def __call__(self, x):
        h0 = max_pooling_2d.max_pooling_2d(relu.relu(self.conv1a(x)), 3, stride=2)
        h1 = max_pooling_2d.max_pooling_2d(relu.relu(self.conv1b(x)), 3, stride=2)
        h0 = max_pooling_2d.max_pooling_2d(relu.relu(self.conv2a(h0)), 3, stride=2)
        h1 = max_pooling_2d.max_pooling_2d(relu.relu(self.conv2b(h1)), 3, stride=2)
        h2 = relu.relu(self.conv3a(h0) + self.conv3b(h1))
        h3 = relu.relu(self.conv3c(h0) + self.conv3d(h1))
        h2 = relu.relu(self.conv4a(h2))
        h3 = relu.relu(self.conv4b(h3))
        h2 = max_pooling_2d.max_pooling_2d(relu.relu(self.conv5a(h2)), 3, stride=2)
        h3 = max_pooling_2d.max_pooling_2d(relu.relu(self.conv5b(h3)), 3, stride=2)
        h3 = self.fc6a(h2) + self.fc6b(h3)
        h3 = self.fc7(h3)
        return self.fc8(h3)
