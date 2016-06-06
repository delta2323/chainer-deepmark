import math
from chainer.functions.activation import relu
from chainer.functions.pooling import average_pooling_2d
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear
from chainer.links.normalization import batch_normalization


class BottleNeckA(link.Chain):
    def __init__(self, in_size, ch, out_size, stride=2, use_cudnn=True):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=convolution_2d.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True, use_cudnn=use_cudnn),
            bn1=batch_normalization.BatchNormalization(ch),
            conv2=convolution_2d.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, use_cudnn=use_cudnn),
            bn2=batch_normalization.BatchNormalization(ch),
            conv3=convolution_2d.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True, use_cudnn=use_cudnn),
            bn3=batch_normalization.BatchNormalization(out_size),
            conv4=convolution_2d.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True, use_cudnn=use_cudnn),
            bn4=batch_normalization.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h1 = relu.relu(self.bn1(self.conv1(x), test=not train))
        h1 = relu.relu(self.bn2(self.conv2(h1), test=not train))
        h1 = self.bn3(self.conv3(h1), test=not train)
        h2 = self.bn4(self.conv4(x), test=not train)

        return relu.relu(h1 + h2)


class BottleNeckB(link.Chain):
    def __init__(self, in_size, ch, use_cudnn=True):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=convolution_2d.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True, use_cudnn=use_cudnn),
            bn1=batch_normalization.BatchNormalization(ch),
            conv2=convolution_2d.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, use_cudnn=use_cudnn),
            bn2=batch_normalization.BatchNormalization(ch),
            conv3=convolution_2d.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True, use_cudnn=use_cudnn),
            bn3=batch_normalization.BatchNormalization(in_size),
        )

    def __call__(self, x, train):
        h = relu.relu(self.bn1(self.conv1(x), test=not train))
        h = relu.relu(self.bn2(self.conv2(h), test=not train))
        h = self.bn3(self.conv3(h), test=not train)

        return relu.relu(h + x)


class Block(link.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2, use_cudnn=True):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride, use_cudnn=use_cudnn))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB(out_size, ch, use_cudnn=use_cudnn))]

        for link in links:
            self.add_link(*link)
        self.forward = links

    def __call__(self, x, train):
        for name,_ in self.forward:
            f = getattr(self, name)
            h = f(x if name == 'a' else h, train)

        return h


class ResNet50(link.Chain):

    insize = 224

    def __init__(self, use_cudnn=True):
        w = math.sqrt(2)
        super(ResNet50, self).__init__(
            conv1=convolution_2d.Convolution2D(3, 64, 7, 2, 3, w, nobias=True, use_cudnn=use_cudnn),
            bn1=batch_normalization.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1, use_cudnn=use_cudnn),
            res3=Block(4, 256, 128, 512, use_cudnn=use_cudnn),
            res4=Block(6, 512, 256, 1024, use_cudnn=use_cudnn),
            res5=Block(3, 1024, 512, 2048, use_cudnn=use_cudnn),
            fc=linear.Linear(2048, 1000),
        )
        self.use_cudnn = use_cudnn
        self.train = True

    def __call__(self, x):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = max_pooling_2d.max_pooling_2d(relu.relu(h), 3, stride=2, use_cudnn=self.use_cudnn)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = average_pooling_2d.average_pooling_2d(h, 7, stride=1, use_cudnn=self.use_cudnn)
        h = self.fc(h)
        return h
