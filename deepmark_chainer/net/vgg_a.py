from chainer.functions.activation import relu
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear


class VGG_A(link.Chain):
    """VGG (network A) without dropout.

    Reference: https://arxiv.org/abs/1409.1556
    
    ``VGG`` assumes its inputs be 4-dimentional tensors of the shape (B, 3, 224, 224).

    """

    def __init__(self, use_cudnn=True):
        super(VGG_A, self).__init__(
            conv1=convolution_2d.Convolution2D(3, 64, 3, use_cudnn=use_cudnn),
            conv2=convolution_2d.Convolution2D(64, 128, 3, use_cudnn=use_cudnn),
            conv3a=convolution_2d.Convolution2D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3b=convolution_2d.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4a=convolution_2d.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4b=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5a=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5b=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            fc6=linear.Linear(25088, 4096),
            fc7=linear.Linear(4096, 4096),
            fc8=linear.Linear(4096, 1000)
        )
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        h = relu.relu(self.conv1(x), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv2(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv3a(h), self.use_cudnn)
        h = relu.relu(self.conv3b(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv4a(h), self.use_cudnn)
        h = relu.relu(self.conv4b(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv5a(h), self.use_cudnn)
        h = relu.relu(self.conv5b(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)
