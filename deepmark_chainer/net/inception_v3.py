from chainer.functions.array import concat
from chainer.functions.loss import softmax_cross_entropy as S
from chainer.functions.noise import dropout
from chainer.functions.pooling import average_pooling_2d as A
from chainer.functions.pooling import max_pooling_2d as M
from chainer import link
from chainer.links.connection import convolution_2d as C
from chainer.links.connection import linear
from chainer.links.normalization import batch_normalization as B


class AuxConv(link.Chain):

    def __init__(self, conv, batch_norm=True, pool=None):
        super(AuxConv, self).__init__(conv=conv)
        if batch_norm:
            out_channel = conv.W.data.shape[0]
            self.add_link('batch_norm',
                          B.BatchNormalization(out_channel))
        self.pool = pool

    def __call__(self, x, train=True):
        if self.pool:
            x = self.pool(x)
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x, test=not train)
        return x


class Sequential(link.ChainList):

    def __call__(self, x, *args, **kwargs):
        for l in self:
            x = l(x, *args, **kwargs)
        return x


class Inception(link.ChainList):

    def __init__(self, *links, **kw):
        super(Inception, self).__init__(*links)
        self.pool = kw.get('pool', None)

    def __call__(self, x, train=True):
        xs = [l(x, train) for l in self]
        if self.pool:
            xs.append(self.pool(x))
        return concat.concat(xs)


class InceptionV3(link.Chain):
    """Inception V3.

    http://arxiv.org/abs/1512.00567
    https://github.com/tensorflow/models/blob/master/inception/inception/slim/inception_model.py

    """

    def __init__(self, use_cudnn=True):
        convolution = link.ChainList(
            AuxConv(C.Convolution2D(3, 32, 3, 2, use_cudnn=use_cudnn)),
            AuxConv(C.Convolution2D(32, 32, 3, use_cudnn=use_cudnn)),
            AuxConv(C.Convolution2D(32, 64, 3, 1, 1, use_cudnn=use_cudnn)),
            AuxConv(C.Convolution2D(64, 80, 3, 1, 1, use_cudnn=use_cudnn)),
            AuxConv(C.Convolution2D(80, 192, 3, use_cudnn=use_cudnn)))

        def inception_0(input_channel, pool_channel):
            # 1x1
            s1 = AuxConv(C.Convolution2D(input_channel, 64, 1, use_cudnn=use_cudnn))

            # 5x5
            s21 = AuxConv(C.Convolution2D(input_channel, 48, 1, use_cudnn=use_cudnn))
            s22 = AuxConv(C.Convolution2D(48, 64, 5, pad=2, use_cudnn=use_cudnn))
            s2 = Sequential(s21, s22)

            # double 3x3
            s31 = AuxConv(C.Convolution2D(input_channel, 64, 1, use_cudnn=use_cudnn))
            s32 = AuxConv(C.Convolution2D(64, 96, 3, pad=1, use_cudnn=use_cudnn))
            s33 = AuxConv(C.Convolution2D(96, 96, 3, pad=1, use_cudnn=use_cudnn))
            s3 = Sequential(s31, s32, s33)

            # pool
            s4 = AuxConv(C.Convolution2D(input_channel,
                                         pool_channel, 3, pad=1, use_cudnn=use_cudnn),
                         pool=M.MaxPooling2D(3, 1, 1, use_cudnn=use_cudnn))

            return Inception(s1, s2, s3, s4)

        inception0 = Sequential(*[inception_0(input_channel, pool_channel)
                                  for input_channel, pool_channel
                                  in zip([192, 256, 288], [32, 64, 64])])

        grid_reduction0 = Inception(
            # strided 3x3
            AuxConv(C.Convolution2D(288, 384, 3, 2, use_cudnn=use_cudnn)),
            # double 3x3
            Sequential(
                AuxConv(C.Convolution2D(288, 64, 1, use_cudnn=use_cudnn)),
                AuxConv(C.Convolution2D(64, 96, 3, pad=1, use_cudnn=use_cudnn)),
                AuxConv(C.Convolution2D(96, 96, 3, 2, use_cudnn=use_cudnn))),
            # pool
            pool=M.MaxPooling2D(3, 2))

        def inception_1(hidden_channel):
            # 1x1
            s1 = AuxConv(C.Convolution2D(768, 192, 1, use_cudnn=use_cudnn))

            # 7x7
            s21 = AuxConv(C.Convolution2D(768, hidden_channel, 1, use_cudnn=use_cudnn))
            s22 = AuxConv(C.Convolution2D(hidden_channel, hidden_channel, (1, 7), pad=(0, 3), use_cudnn=use_cudnn))
            s23 = AuxConv(C.Convolution2D(hidden_channel, 192, (7, 1), pad=(3, 0), use_cudnn=use_cudnn))
            s2 = Sequential(s21, s22, s23)

            # double 7x7
            s31 = AuxConv(C.Convolution2D(768, hidden_channel, 1, use_cudnn=use_cudnn))
            s32 = AuxConv(C.Convolution2D(hidden_channel, hidden_channel, (1, 7), pad=(0, 3), use_cudnn=use_cudnn))
            s33 = AuxConv(C.Convolution2D(hidden_channel, hidden_channel, (7, 1), pad=(3, 0), use_cudnn=use_cudnn))
            s34 = AuxConv(C.Convolution2D(hidden_channel, hidden_channel, (1, 7), pad=(0, 3), use_cudnn=use_cudnn))
            s35 = AuxConv(C.Convolution2D(hidden_channel, 192, (7, 1), pad=(3, 0), use_cudnn=use_cudnn))
            s3 = Sequential(s31, s32, s33, s34, s35)

            # pool
            s4 = AuxConv(C.Convolution2D(768, 192, 3, pad=1, use_cudnn=use_cudnn),
                         pool=A.AveragePooling2D(3, 1, 1, use_cudnn=use_cudnn))

            return Inception(s1, s2, s3, s4)

        inception1 = Sequential(*[inception_1(c)
                                  for c in [128, 160, 160, 192]])

        grid_reduction1 = Inception(
            # strided 3x3
            Sequential(
                AuxConv(C.Convolution2D(768, 192, 1, use_cudnn=use_cudnn)),
                AuxConv(C.Convolution2D(192, 320, 3, 2, use_cudnn=use_cudnn))),
            # 7x7 and 3x3
            Sequential(
                AuxConv(C.Convolution2D(768, 192, 1, use_cudnn=use_cudnn)),
                AuxConv(C.Convolution2D(192, 192, (1, 7), pad=(0, 3), use_cudnn=use_cudnn)),
                AuxConv(C.Convolution2D(192, 192, (7, 1), pad=(3, 0), use_cudnn=use_cudnn)),
                AuxConv(C.Convolution2D(192, 192, 3, 2, use_cudnn=use_cudnn))),
            # pool
            pool=M.MaxPooling2D(3, 2, use_cudnn=use_cudnn))

        def inception_2(input_channel):
            # 1x1
            s1 = AuxConv(C.Convolution2D(input_channel, 320, 1, use_cudnn=use_cudnn))

            # 3x3
            s21 = AuxConv(C.Convolution2D(input_channel, 384, 1, use_cudnn=use_cudnn))
            s22 = Inception(AuxConv(C.Convolution2D(384, 384, (1, 3),
                                                    pad=(0, 1), use_cudnn=use_cudnn)),
                            AuxConv(C.Convolution2D(384, 384, (3, 1),
                                                    pad=(1, 0), use_cudnn=use_cudnn)))
            s2 = Sequential(s21, s22)

            # double 3x3
            s31 = AuxConv(C.Convolution2D(input_channel, 448, 1, use_cudnn=use_cudnn))
            s32 = AuxConv(C.Convolution2D(448, 384, 3, pad=1, use_cudnn=use_cudnn))
            s331 = AuxConv(C.Convolution2D(384, 384, (1, 3), pad=(0, 1), use_cudnn=use_cudnn))
            s332 = AuxConv(C.Convolution2D(384, 384, (3, 1), pad=(1, 0), use_cudnn=use_cudnn))
            s33 = Inception(s331, s332)
            s3 = Sequential(s31, s32, s33)

            # pool
            s4 = AuxConv(C.Convolution2D(input_channel, 192, 3, pad=1, use_cudnn=use_cudnn),
                         pool=A.AveragePooling2D(3, 1, 1, use_cudnn=use_cudnn))
            return Inception(s1, s2, s3, s4)

        inception2 = Sequential(*[inception_2(input_channel)
                                  for input_channel in [1280, 2048]])

        auxiliary_convolution = Sequential(
            AuxConv(C.Convolution2D(768, 128, 1, use_cudnn=use_cudnn),
                    pool=A.AveragePooling2D(5, 3, use_cudnn=use_cudnn)),
            AuxConv(C.Convolution2D(128, 768, 5, use_cudnn=use_cudnn)))

        super(InceptionV3, self).__init__(
            convolution=convolution,
            inception=link.ChainList(inception0, inception1, inception2),
            grid_reduction=link.ChainList(grid_reduction0, grid_reduction1),
            auxiliary_convolution=auxiliary_convolution,
            auxiliary_linear=linear.Linear(768, 1000),
            linear=linear.Linear(2048, 1000))

    def __call__(self, x, train=True):
        """Computes the output of the module.

        Args:
            x(~chainer.Variable): Input variable.

        """

        def convolution(x, train):
            x = self.convolution[0](x)
            x = self.convolution[1](x)
            x = self.convolution[2](x)
            x = M.max_pooling_2d(x, 3, 2)
            x = self.convolution[3](x)
            x = self.convolution[4](x)
            x = M.max_pooling_2d(x, 3, 2)
            return x

        # Original paper and TensorFlow implementation has different
        # auxiliary classifier. We implement latter one.
        def auxiliary_classifier(x, train):
            x = self.auxiliary_convolution(x, train)
            return self.auxiliary_linear(x)

        def classifier(x, train):
            x = A.average_pooling_2d(x, 8)
            x = dropout.dropout(x, train=train)
            x = self.linear(x)
            return x

        x = convolution(x, train)
        assert x.data.shape[1:] == (192, 35, 35),\
            'actual={}'.format(x.data.shape[1:])
        x = self.inception[0](x, train)
        assert x.data.shape[1:] == (288, 35, 35),\
            'actual={}'.format(x.data.shape[1:])
        x = self.grid_reduction[0](x, train)
        assert x.data.shape[1:] == (768, 17, 17),\
            'actual={}'.format(x.data.shape[1:])
        x = self.inception[1](x, train)
        assert x.data.shape[1:] == (768, 17, 17),\
            'actual={}'.format(x.data.shape[1:])
        y_aux = auxiliary_classifier(x, train)
        x = self.grid_reduction[1](x, train)
        assert x.data.shape[1:] == (1280, 8, 8),\
            'actual={}'.format(x.data.shape[1:])
        x = self.inception[2](x, train)
        assert x.data.shape[1:] == (2048, 8, 8),\
            'actual={}'.format(x.data.shape[1:])
        y = classifier(x, train)
        self.y = y
        self.y_aux = y_aux
        return y, y_aux


class InceptionV3Classifier(link.Chain):

    def __init__(self, predictor):
        super(InceptionV3Classifier, self).__init__(predictor=predictor)

    def __call__(self, *args):

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y, self.y_aux = self.predictor(*x)
        self.loss = S.softmax_cross_entropy(self.y, t)
        self.loss += S.softmax_cross_entropy(self.y_aux, t)
        return self.loss
