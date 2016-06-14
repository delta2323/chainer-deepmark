import six

from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer import link
from chainer.links.connection import convolution_2d as C
from chainer.links.connection import linear as L
from chainer.links.connection import lstm
from chainer.links.connection import gru
from chainer.links.normalization import batch_normalization as B


class BRNN(link.Chain):

    def __init__(self, input_dim, output_dim, rnn_unit):
        if rnn_unit == 'LSTM':
            forward = lstm.LSTM(input_dim, output_dim)
            reverse = lstm.LSTM(input_dim, output_dim)
        elif rnn_unit == 'GRU':
            forward = gru.StatefulGRU(output_dim, input_dim)
            reverse = gru.StatefulGRU(output_dim, input_dim)
        super(BRNN, self).__init__(forward=forward, reverse=reverse)

    def reset_state(self):
        self.forward.reset_state()
        self.reverse.reset_state()

    def __call__(self, xs, train=True):
        N = len(xs)
        x_forward = [self.forward(x, train) for x in xs]
        x_reverse = [self.reverse(xs[n], train) for n
                     in six.moves.range(N - 1, -1, -1)]
        x_reverse.reverse()
        return [x_f + x_r for x_f, x_r in zip(x_forward, x_reverse)]


class ConvBN(link.Chain):

    def __init__(self, *args, **kwargs):
        conv = C.Convolution2D(*args, **kwargs)
        out_channel = conv.W.data.shape[0]
        batch_norm = B.BatchNormalization(out_channel)
        super(ConvBN, self).__init__(conv=conv, batch_norm=batch_norm)

    def __call__(self, x, train=True):
        x = self.conv(x)
        return self.batch_norm(x, test=not train)


class LinearBN(link.Chain):

    def __init__(self, *args, **kwargs):
        linear = L.Linear(*args, **kwargs)
        out_channel = len(linear.W.data)
        batch_norm = B.BatchNormalization(out_channel)
        super(LinearBN, self).__init__(linear=linear, batch_norm=batch_norm)

    def __call__(self, x, train=True):
        x = self.linear(x)
        return self.batch_norm(x, test=not train)


class Sequential(link.ChainList):

    def __call__(self, x, *args, **kwargs):
        for l in self:
            x = l(x, *args, **kwargs)
        return x


class DeepSpeech2(link.Chain):

    def __init__(self, channel_dim=32, hidden_dim=1760, out_dim=29, rnn_unit='Linear', use_cudnn=True):
        c1 = ConvBN(1, channel_dim, (5, 20), 2, use_cudnn=use_cudnn)
        c2 = ConvBN(channel_dim, channel_dim, (5, 10), (1, 2), use_cudnn=use_cudnn)
        convolution = Sequential(c1, c2)

        brnn1 = BRNN(31 * channel_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn2 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn3 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn4 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn5 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn6 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn7 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        recurrent = Sequential(brnn1, brnn2, brnn3, brnn4,
                               brnn5, brnn6, brnn7)

        fc1 = LinearBN(hidden_dim, hidden_dim)
        fc2 = L.Linear(hidden_dim, out_dim)
        linear = link.ChainList(fc1, fc2)
        super(DeepSpeech2, self).__init__(convolution=convolution,
                                          recurrent=recurrent,
                                          linear=linear)

    def _linear(self, xs, train=True):
        ret = []
        for x in xs:
            x = self.linear[0](x, train)
            x = self.linear[1](x)
            ret.append(x)
        return ret

    def __call__(self, x, train=True):
        x = reshape.reshape(x, (len(x.data), 1) + x.data.shape[1:])
        x = self.convolution(x, train)
        xs = split_axis.split_axis(x, x.data.shape[2], 2)
        for x in xs:
            x.data = self.xp.ascontiguousarray(x.data)
        for r in self.recurrent:
            r.reset_state()
        xs = self.recurrent(xs, train)
        xs = self._linear(xs, train)
        return xs
