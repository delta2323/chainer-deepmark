#!/usr/bin/env python

import sys

from setuptools import setup


setup(
    name='deepmark_chainer',
    version='0.0.1',
    packages=['deepmark_chainer',
              'deepmark_chainer.net',
              'deepmark_chainer.utils'],
    setup_requires=[],
    install_requires=['chainer'],
    tests_require=['mock',
                   'nose']
)
