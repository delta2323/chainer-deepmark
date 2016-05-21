import argparse

import chainer
from chainer import links as L
from chainer import optimizers as O
from chainer import cuda
from chainer import function_hooks
import numpy
import six


parser = argparse.ArgumentParser(description='ConvNet benchmark')
parser.add_argument('--model', '-m', type=str, default='inception-v3-batchnorm',
                    choices=('inception-v3-batchnorm', 'alexnet-owt', 'vgg', 'resnet-50'),
                    help='network architecture')
parser.add_argument('--iteration', '-i', type=int, default=10, help='iteration')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU to use. Negative value to use CPU')
parser.add_argument('--cudnn', '-c', action='store_true', help='True if using cudnn')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='batchsize')
args = parser.parse_args()


if args.model == 'inception-v3-batchnorm':
    model = L.InceptionV3()
elif args.model == 'alex':
    model = L.Alex()
elif args.model == 'vgg':
    model = L.VGG()
elif args.model == 'resnet-50':
    model = L.Resnet(50)
else:
    raise ValueError('Invalid model name:{}'.format(args.model))

in_size = 10
out_size = 3

net = L.Linear(in_size, out_size)
model = L.Classifier(net)

optimizer = O.SGD()
optimizer.setup(model)


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = cuda.cupy if args.gpu >= 0 else numpy


data = numpy.random.uniform(-1, 1, (args.batchsize, in_size)).astype(numpy.float32)
data.fill(33333)
data = chainer.Variable(xp.asarray(data))
label = numpy.random.randint(0, out_size, (args.batchsize,)).astype(dtype=numpy.int32)
label = chainer.Variable(xp.asarray(label))

timer = function_hooks.AccumulateTimerHook(xp)
print('iteration\tforward\tbackward\tupdate')
for iteration in six.moves.range(args.iteration):
    with timer('forward'):
        loss = model(data, label)

    with timer('backward'):
        loss.backward()

    with timer('update'):
        optimizer.update()

    print('{}\t{}\t{}\t{}'.format(iteration,
                                  timer.last_increment('forward'),
                                  timer.last_increment('backward'),
                                  timer.last_increment('update')))

print('forward:{}'.format(timer.total_time('forward') / args.iteration))
print('backward:{}'.format(timer.total_time('backward') / args.iteration))
print('update:{}'.format(timer.total_time('update') / args.iteration))
