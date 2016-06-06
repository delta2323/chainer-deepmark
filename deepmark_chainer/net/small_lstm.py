from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions.array import reshape
from chainer import link
from chainer.links.connection import embed_id
from chainer.links.connection import lstm
from chainer.links.connection import gru
from chainer.links.connection import linear as linear_


class SmallLSTM(link.Chain):
    """

    https://github.com/karpathy/char-rnn/blob/master/train.lua#L38-L48

    """

    def __init__(self, vocab_size=10, unit_size=128, rnn_unit='LSTM'):
        embed = embed_id.EmbedID(vocab_size, unit_size)
        if rnn_unit == 'LSTM':
            rnn1 = lstm.LSTM(unit_size, unit_size)
            rnn2 = lstm.LSTM(unit_size, unit_size)
        elif rnn_unit == 'GRU':
            rnn1 = gru.StatefulGRU(unit_size, unit_size)
            rnn2 = gru.StatefulGRU(unit_size, unit_size)
        else:
            raise ValueError('Invalid RNN unit:{}'.format(rnn_unit))
        linear = linear_.Linear(unit_size, vocab_size)

        super(SmallLSTM, self).__init__(embed=embed,
                                        rnn1=rnn1, rnn2=rnn2,
                                        linear=linear)

    def reset_state(self):
        self.rnn1.reset_state()
        self.rnn2.reset_state()

    def __call__(self, x):
        x = self.embed(x)
        xs = split_axis.split_axis(x, x.data.shape[1], 1)
        ret = []
        for x in xs:
            x = self.rnn1(x)
            x = self.rnn2(x)
            x = self.linear(x)
            x = reshape.reshape(x, x.data.shape + (-1,))
            ret.append(x)
        ret = concat.concat(ret, axis=2)
        return ret
