from chainer import cuda
from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear
from chainer import variable
from chainer.utils import conv


def _triplet(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x, x


class Convolution3D(link.Link):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, use_cudnn=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = _triplet(ksize)
        self.stride = _triplet(stride)
        self.pad = _triplet(pad)
        super(Convolution3D, self).__init__()

    def __call__(self, x):
        xp = cuda.get_array_module(x.data)
        l = conv.get_conv_outsize(x.data.shape[2], self.ksize[0], self.stride[0], self.pad[0])
        h = conv.get_conv_outsize(x.data.shape[3], self.ksize[1], self.stride[1], self.pad[1])
        w = conv.get_conv_outsize(x.data.shape[4], self.ksize[2], self.stride[2], self.pad[2])
        shape = (len(x.data), self.out_channels, l, h, w)
        return variable.Variable(xp.random.uniform(-1, 1, shape).astype(x.data.dtype),
                                 volatile=x.volatile)


def max_pooling_3d(x, ksize, stride=None, pad=0, use_cudnn=True):
    xp = cuda.get_array_module(x.data)
    if stride is None:
        stride = ksize
    ksize = _triplet(ksize)
    stride = _triplet(stride)
    pad = _triplet(pad)
    l = conv.get_conv_outsize(x.data.shape[2], ksize[0], stride[0], pad[0])
    h = conv.get_conv_outsize(x.data.shape[3], ksize[1], stride[1], pad[1])
    w = conv.get_conv_outsize(x.data.shape[4], ksize[2], stride[2], pad[2])
    shape = (len(x.data), x.data.shape[1], l, h, w)
    return variable.Variable(xp.random.uniform(-1, 1, shape).astype(x.data.dtype),
                             volatile=x.volatile)


class C3D(link.Chain):

    """C3D

    https://github.com/facebook/C3D/blob/master/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_frm.prototxt

    """

    def __init__(self, use_cudnn=True):
        super(C3D, self).__init__(
            conv1a=Convolution3D(3, 64, 3, pad=1, use_cudnn=use_cudnn),
            conv2a=Convolution3D(64, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv3a=Convolution3D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3b=Convolution3D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4a=Convolution3D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4b=Convolution3D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5a=Convolution3D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5b=Convolution3D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            fc6=linear.Linear(4608, 4096),
            fc7=linear.Linear(4096, 4096),
            fc8=linear.Linear(4096, 487))
        self.use_cudnn = use_cudnn


    def __call__(self, x):
        x = relu.relu(self.conv1a(x))
        x = max_pooling_3d(x, (1, 2, 2), use_cudnn=self.use_cudnn)
        x = relu.relu(self.conv2a(x))
        x = max_pooling_3d(x, 2, use_cudnn=self.use_cudnn)
        x = relu.relu(self.conv3a(x))
        x = relu.relu(self.conv3b(x))
        x = max_pooling_3d(x, 2, use_cudnn=self.use_cudnn)
        x = relu.relu(self.conv4a(x))
        x = relu.relu(self.conv4b(x))
        x = max_pooling_3d(x, 2, use_cudnn=self.use_cudnn)
        x = relu.relu(self.conv5a(x))
        x = relu.relu(self.conv5b(x))
        x = max_pooling_3d(x, 2, use_cudnn=self.use_cudnn)
        x = relu.relu(self.fc6(x))
        x = relu.relu(self.fc7(x))
        return self.fc8(x)
