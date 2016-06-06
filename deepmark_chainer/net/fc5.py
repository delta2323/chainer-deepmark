from chainer import link
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer.links.connection import linear


class FC5(link.ChainList):
    """

    https://github.com/Alexey-Kamenev/Benchmarks/blob/master/CNTK/ffn.config

    """

    def __init__(self, in_size=512, hidden_size=2048, out_size=10000):
        super(FC5, self).__init__(linear.Linear(in_size, hidden_size),
                                  linear.Linear(hidden_size, hidden_size),
                                  linear.Linear(hidden_size, hidden_size),
                                  linear.Linear(hidden_size, hidden_size),
                                  linear.Linear(hidden_size, out_size))

    def __call__(self, x):
        xs = split_axis.split_axis(x, x.data.shape[1], 1)
        ret = []
        for x in xs:
            for l in self:
                x = l(x)
            ret.append(x)
        return ret
