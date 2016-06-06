import os
import shutil

from chainer import cuda


def clear_cache(cache_level):
    if cache_level is None:
        cache_dir = cuda.cupy.cuda.get_cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    if cache_level is None or cache_level == 'disk':
        cuda.cupy.clear_memo()

