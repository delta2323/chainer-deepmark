import argparse

from chainer import cuda
from chainer import links as L
from chainer import optimizers as O
import numpy
import six

from deepmark_chainer import net
from deepmark_chainer.utils import cache
from deepmark_chainer.utils import timer


parser = argparse.ArgumentParser(
    description='Deepmark benchmark for image data.')
parser.add_argument('--predictor', '-p', type=str, default='inception-v3',
                    choices=('inception-v3', 'alex-owt', 'vgg-d', 'resnet-50'),
                    help='Network architecture')
parser.add_argument('--seed', '-s', type=int, default=0,
                    help='Random seed')
parser.add_argument('--iteration', '-i', type=int, default=5,
                    help='The number of iteration to be averaged over.')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU to use. Negative value to use CPU')
parser.add_argument('--cudnn', '-c', action='store_true',
                    help='If this flag is set, cuDNN is enabled.')
parser.add_argument('--dry-run', '-d', type=int, default=5,
                    help='The number of iterations of a dry run '
                    'not counted towards final timing')
parser.add_argument('--workspace-ratio', '-w', type=float, default=0.1,
                    help='This option determins workspace size of cuDNN. '
                    'By default, 10 precent of total GPU memory is used for cuDNN\'s workspace. '
                    'You may see some speed-up by increasing this ratio, '
                    'while you may train a network with larger batch size by decreasing the ratio. '
                    'Note that the option gets effective only when GPU is used.' )
parser.add_argument('--cache-level', '-C', type=str, default='none',
                    choices=('none', 'memory', 'disk'),
                    help='This option determines the type of the kernel '
                    'cache used.'
                    'By default, memory cache and disk cache are removed '
                    'at the beginning of every iteration. '
                    'Otherwise, elapsed times of each iteration are '
                    'measured with corresponding cache enabled. ')
parser.add_argument('--batchsize', '-b', type=int, default=32,
                    help='Batchsize')
args = parser.parse_args()
print(args)

numpy.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)

if args.gpu >= 0:
    if args.workspace_ratio < 0.0 or args.workspace_ratio > 1.0:
        raise ValueError('Invalid workspace ratio:{}  (valid interval:[0.0,1.0])'.format(args.workspace_ratio))
    _free_mem, total_mem = cuda.cupy.cuda.runtime.memGetInfo()
    size = long(total_mem * args.workspace_ratio)
    cuda.set_max_workspace_size(size)

in_channels = 3
label_num = 100

if args.predictor == 'inception-v3':
    predictor = net.inception_v3.InceptionV3(use_cudnn=args.cudnn)
    model = net.inception_v3.InceptionV3Classifier(predictor)
    in_size = 299
elif args.predictor == 'alex-owt':
    predictor = net.alex_owt.AlexOWT(use_cudnn=args.cudnn)
    model = L.Classifier(predictor)
    in_size = 224
elif args.predictor == 'vgg-d':
    predictor = net.vgg_d.VGG_D(use_cudnn=args.cudnn)
    model = L.Classifier(predictor)
    in_size = 224
elif args.predictor == 'resnet-50':
    predictor = net.resnet_50.ResNet50(use_cudnn=args.cudnn)
    model = L.Classifier(predictor)
    in_size = 224
else:
    raise ValueError('Invalid architector:{}'.format(args.predictor))


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
optimizer = O.SGD(0.0001)
optimizer.use_cleargrads()
optimizer.setup(model)

xp = cuda.cupy if args.gpu >= 0 else numpy


# data generation
total_iteration = args.dry_run + args.iteration
shape = (total_iteration, args.batchsize, in_channels, in_size, in_size)
data = xp.random.uniform(-1, 1, shape)
data = data.astype(numpy.float32, copy=False)
label = xp.random.randint(0, label_num, (total_iteration, args.batchsize))
label = label.astype(numpy.int32, copy=False)

# dry run
for iteration in six.moves.range(args.dry_run):
    print('Dry Run\t{}'.format(iteration))

    loss = model(data[iteration], label[iteration])
    model.cleargrads()
    loss.backward()
    optimizer.update()

# evaluation
with timer.get_timer(xp) as t:
    for iteration in six.moves.range(args.iteration):
        print('Iteration\t{}'.format(iteration))
        if args.gpu >= 0:
            cache.clear_cache(args.cache_level)

        idx = iteration + args.dry_run
        loss = model(data[idx], label[idx])
        model.cleargrads()
        loss.backward()
        optimizer.update()


time_taken_per_iter = t.total_time() / args.iteration
examples_per_sec = 1 / time_taken_per_iter * args.batchsize

if args.gpu > 0:
    device = 'GPU{}'.format(args.gpu)
    if args.cudnn:
        backend = 'cuDNN'
    else:
        backend = 'CuPy'
else:
    device = 'CPU'
    backend = 'NumPy'
print('Device:{}\tNetwork:{}\tBackend:{}\t'
      'Batchsize:{}\tIter (ms):{}\t'
      'Examples/sec:{}'.format(device, args.predictor,
                               backend, args.batchsize,
                               time_taken_per_iter * 1000,
                               examples_per_sec))
