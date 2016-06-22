import unittest

import numpy

import chainer
from chainer import cuda
from chainer.testing import attr

from deepmark_chainer.net import vgg


class TestVGG(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 3, 224, 224)).astype(numpy.float32)
        self.l = vgg.VGG()

    def check_forward(self, xp):
        x = chainer.Variable(xp.asarray(self.x))
        self.l(x)

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.l.to_gpu()
        self.check_forward(cuda.cupy)
