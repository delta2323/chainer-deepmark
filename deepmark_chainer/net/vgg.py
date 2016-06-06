from chainer.functions.activation import relu
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear


class VGG(link.Chain):

    """VGG(network A) with dropout removed."""

    def __init__(self, use_cudnn):
        super(VGG, self).__init__(
            conv1=convolution_2d.Convolution2D(3, 64, 3, use_cudnn=use_cudnn),
            conv2=convolution_2d.Convolution2D(64, 256, 3, use_cudnn=use_cudnn),
            conv3=convolution_2d.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4=convolution_2d.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv5=convolution_2d.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv6=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv7=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv8=convolution_2d.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
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
        h = relu.relu(self.conv3(h), self.use_cudnn)
        h = relu.relu(self.conv4(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv5(h), self.use_cudnn)
        h = relu.relu(self.conv6(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = relu.relu(self.conv7(h), self.use_cudnn)
        h = relu.relu(self.conv8(h), self.use_cudnn)
        h = max_pooling_2d.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)
