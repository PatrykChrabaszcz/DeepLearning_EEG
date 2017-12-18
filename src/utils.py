import time
import glob
import os
import numpy as np
from torch.autograd import Variable
import torch


class Stats:
    def __init__(self, message=''):
        self.s_t = None
        self.message = message

    def __enter__(self):
        self.s_t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('%s : %f' % (self.message, time.time()-self.s_t))
