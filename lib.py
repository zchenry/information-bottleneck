import warnings
warnings.filterwarnings("ignore")

import os
import pdb
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.iterators import SerialIterator
from chainer import initializers
from chainer.cuda import to_cpu

import multiprocessing
from joblib import Parallel, delayed
