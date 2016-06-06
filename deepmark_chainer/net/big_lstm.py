from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.functions.array import reshape
from chainer.functions.noise import dropout
from chainer import link
from chainer.links.connection import embed_id
from chainer.links.connection import lstm
from chainer.links.connection import gru
from chainer.links.connection import linear


class BigLSTM(link.Chain):
    """

    http://arxiv.org/abs/1602.02410v2
    
    """

    def __init__(self, vocab_size=10, rnn_unit='LSTM'):
        embed = embed_id.EmbedID(vocab_size, 10)
        if rnn_unit == 'LSTM':
            rnns = link.ChainList(lstm.LSTM(10, 20),
                                  lstm.LSTM(20, 20))
        elif rnn_unit == 'GRU':
            rnns = link.ChainList(gru.StatefulGRU(20, 10),
                                  gru.StatefulGRU(20, 20))
        else:
            raise ValueError('Invalid RNN unit:{}'.format(rnn_unit))

        linears = link.ChainList(linear.Linear(20, 10),
                                 linear.Linear(10, vocab_size))
        super(BigLSTM, self).__init__(embed=embed, rnns=rnns,
                                      linears=linears)
        self.train = True
        # initialize the bias vector of forget gates by 1.0.

    def reset_state(self):
        for l in self.rnns:
            l.reset_state()

    def __call__(self, x):
        x = self.embed(x)
        xs = split_axis.split_axis(x, x.data.shape[1], 1)
        ret = []
        for x in xs:
            for l in self.rnns:
                x = l(x)
                x = dropout.dropout(x, 0.25, self.train)
            for l in self.linears:
                x = l(x)
            x = reshape.reshape(x, x.data.shape + (-1,))
            ret.append(x)
        ret = concat.concat(ret, axis=2)
        return ret
