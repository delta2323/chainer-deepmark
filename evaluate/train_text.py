import argparse

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers as O
from chainer import cuda
import numpy
import six

from deepmark_chainer import net
from deepmark_chainer.utils import timer
from deepmark_chainer.utils import cache


parser = argparse.ArgumentParser(description='Deepmark benchmark for text data.')
parser.add_argument('--predictor', '-p', type=str, default='big-lstm',
                    choices=('small-lstm', 'big-lstm'),
                    help='Network architecture')
parser.add_argument('--seed', '-s', type=int, default=0,
                    help='Random seed')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='The number of iteration to be averaged over.')
parser.add_argument('--seq-length', '-t', type=int, default=200,
                    help='Sequence length')
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
parser.add_argument('--vocab-size', type=int, default=10,
                    help='Number of vocablaries.')
parser.add_argument('--batchsize', '-b', type=int, default=50, help='Batchsize')
args = parser.parse_args()

numpy.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)


if args.predictor == 'small-lstm':
    predictor = net.small_lstm.SmallLSTM(args.vocab_size)
elif args.predictor == 'big-lstm':
    predictor = net.big_lstm.BigLSTM(args.vocab_size)
else:
    raise ValueError('Invalid architector:{}'.format(args.predictor))
model = L.Classifier(predictor)
model.compute_accuracy = False

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
    data = numpy.random.randint(0, args.vocab_size,
                                (args.batchsize, args.seq_length)
                                ).astype(numpy.int32)
    data = chainer.Variable(xp.asarray(data))
    label = numpy.random.randint(0, args.vocab_size,
                                 (args.batchsize, args.seq_length)
                                 ).astype(numpy.int32)
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
