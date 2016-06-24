from chainer.functions.activation import relu
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear


class VGG_D(link.Chain):
    """VGG (network D) without dropout.

    Reference: https://arxiv.org/abs/1409.1556

    ``VGG`` assumes its inputs be 4-dimentional tensors of the shape (B, 3, 224, 224).

    """

    def __init__(self, use_cudnn=True):
        super(VGG_D, self).__init__(
            conv1_1=convolution_2d.Convolution2D(3, 64, 3, pad=1, use_cudnn=use_cudnn),
            conv1_2=convolution_2d.Convolution2D(64, 64, 3, pad=1, use_cudnn=use_cudnn),
            conv2_1=convolution_2d.Convolution2D(64, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv2_2=convolution_2d.Convolution2D(128, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv3_1=convolution_2d.Convolution2D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_2=convolution_2d.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_3=convolution_2d.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4_1=convolution_2d.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_2=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_3=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_1=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_2=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_3=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            fc6=linear.Linear(512 * 7 * 7, 4096),
            fc7=linear.Linear(4096, 4096),
            fc8=linear.Linear(4096, 1000)
        )
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        h = relu.relu(self.conv1_1(x), self.use_cudnn)
        h = relu.relu(self.conv1_2(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv2_1(h), self.use_cudnn)
        h = relu.relu(self.conv2_2(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv3_1(h), self.use_cudnn)
        h = relu.relu(self.conv3_2(h), self.use_cudnn)
        h = relu.relu(self.conv3_3(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv4_1(h), self.use_cudnn)
        h = relu.relu(self.conv4_2(h), self.use_cudnn)
        h = relu.relu(self.conv4_3(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv5_1(h), self.use_cudnn)
        h = relu.relu(self.conv5_2(h), self.use_cudnn)
        h = relu.relu(self.conv5_3(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)
