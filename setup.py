#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


NAME = 'nn4post'
DESCRIPTION = 'Neural network for posterior'
AUTHOR = 'shuiruge'
AUTHOR_EMAIL = 'shuiruge@hotmail.com'
URL = 'https://github.com/shuiruge/nn4post/'
VERSION = '0.9.0'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
 
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license='GPL',
    url=URL,
    packages=find_packages(exclude=[
        'tests.*', 'tests',
        'examples.*', 'examples',
        'dat.*', 'dat',
    ]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Variational Inference'
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPL License',
        'Operating System :: OS Independent',
        'Programming Language :: Python 3+',
        'Framework :: TensorFlow',
    ],
    zip_safe=False,
)
