import unittest

import numpy

import chainer
from chainer import cuda
from chainer.testing import attr

from deepmark_chainer.net import deepspeech2

class TestDeepSpeech2(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 200, 161)).astype(numpy.float32)
        self.l = deepspeech2.DeepSpeech2()

    def check_forward(self, xp):
        x = chainer.Variable(xp.asarray(self.x))
        self.l(x)

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.l.to_gpu()
        self.check_forward(cuda.cupy)

