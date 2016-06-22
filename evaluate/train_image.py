import argparse

import chainer
from chainer import links as L
from chainer import optimizers as O
from chainer import cuda
import numpy
import six

from deepmark_chainer import net
from deepmark_chainer.utils import timer
from deepmark_chainer.utils import cache


parser = argparse.ArgumentParser(description='Deepmark benchmark for image data.')
parser.add_argument('--predictor', '-p', type=str, default='inception-v3',
                    choices=('inception-v3', 'alex-owt', 'vgg', 'resnet-50'),
                    help='Network architecture')
parser.add_argument('--seed', '-s', type=int, default=0,
                    help='Random seed')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='The number of iteration to be averaged over.')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU to use. Negative value to use CPU')
parser.add_argument('--cudnn', '-c', action='store_true', help='If this flag is set, cuDNN is enabled.')
parser.add_argument('--cache-level', '-C', type=str, default='none',
                    choices=('none', 'memory', 'disk'),
                    help='This option determines the type of the kernel cache used.'
                    'By default, memory cache and disk cache are removed '
                    'at the beginning of every iteration. '
                    'Otherwise, elapsed times of each iteration are '
                    'measured with corresponding cache enabled. '
                    'If either cache is enabled, this script operates one additional '
                    'iteration for burn-in before measurement. '
                    'This iteration is not included in the mean elapsed time.'
                    'If we do not use GPU, we do not clear cache at all regardless of the value of '
                    'this option.')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Batchsize')
args = parser.parse_args()

numpy.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)


in_channels = 3

if args.predictor == 'inception-v3':
    predictor = net.inception_v3.InceptionV3(use_cudnn=args.cudnn)
    in_size = 299
elif args.predictor == 'alex-owt':
    predictor = net.alex_owt.AlexOWT(use_cudnn=args.cudnn)
    in_size = 224
elif args.predictor == 'vgg':
    predictor = net.vgg.VGG(use_cudnn=args.cudnn)
    in_size = 224
elif args.predictor == 'resnet-50':
    predictor = net.resnet_50.ResNet50(use_cudnn=args.cudnn)
    in_size = 224
else:
    raise ValueError('Invalid architector:{}'.format(args.predictor))
model = L.Classifier(predictor)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
optimizer = O.SGD()
optimizer.setup(model)

xp = cuda.cupy if args.gpu >= 0 else numpy

start_iteration = 0 if args.cache_level is None else -1

forward_time = 0.0
backward_time = 0.0
update_time = 0.0
print('iteration\tforward\tbackward\tupdate (in seconds)')
for iteration in six.moves.range(start_iteration, args.iteration):
    if args.gpu >= 0:
        cache.clear_cache(args.cache_level)

    # data generation
    data = numpy.random.uniform(-1, 1,
                                (args.batchsize, in_channels, in_size, in_size)).astype(numpy.float32)
    data = chainer.Variable(xp.asarray(data))
    label = numpy.zeros((args.batchsize,), dtype=numpy.int32)
    label = chainer.Variable(xp.asarray(label))

    # forward
    with timer.get_timer(xp) as t:
        loss = model(data, label)
    forward_time_one = t.total_time()

    # backward
    with timer.get_timer(xp) as t:
        loss.backward()
    backward_time_one = t.total_time()

    # parameter update
    with timer.get_timer(xp) as t:
        optimizer.update()
    update_time_one = t.total_time()

    if iteration < 0:
        print('Burn-in\t{}\t{}\t{}'.format(forward_time_one, backward_time_one, update_time_one))
    else:
        print('{}\t{}\t{}\t{}'.format(iteration, forward_time_one, backward_time_one, update_time_one))
        forward_time += forward_time_one
        backward_time += backward_time_one
        update_time += update_time_one

forward_time /= args.iteration
backward_time /= args.iteration
update_time /= args.iteration

print('Mean\t{}\t{}\t{}'.format(forward_time, backward_time, update_time))
