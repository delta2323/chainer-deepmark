from chainer import link
from chainer.functions.activation import relu
from chainer.functions.pooling import max_pooling_2d
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear


class Alex(link.Chain):

    """AlexNet with normalization and dropout removed.

    AlexNet assumes its inputs be 4-dimentional tensors of the shape (B, 3, 224, 224).

    """

    def __init__(self, use_cudnn=True):
        super(Alex, self).__init__(
            conv1=convolution_2d.Convolution2D(3, 96, 11, stride=4, pad=2,
                                               use_cudnn=use_cudnn),
            conv2=convolution_2d.Convolution2D(96, 256, 5, pad=2, use_cudnn=use_cudnn),
            conv3=convolution_2d.Convolution2D(256, 384, 3, pad=1, use_cudnn=use_cudnn),
            conv4=convolution_2d.Convolution2D(384, 384, 3, pad=1, use_cudnn=use_cudnn),
            conv5=convolution_2d.Convolution2D(384, 256, 3, pad=1, use_cudnn=use_cudnn),
            fc6=linear.Linear(9216, 4096),
            fc7=linear.Linear(4096, 4096),
            fc8=linear.Linear(4096, 1000))

    def __call__(self, x):
        h = max_pooling_2d.max_pooling_2d(relu.relu(self.conv1(x)), 3, stride=2)
        h = max_pooling_2d.max_pooling_2d(relu.relu(self.conv2(h)), 3, stride=2)
        h = relu.relu(self.conv3(h))
        h = relu.relu(self.conv4(h))
        h = max_pooling_2d.max_pooling_2d(relu.relu(self.conv5(h)), 3, stride=2)
        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)
