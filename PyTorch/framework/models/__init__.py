# -*- coding: utf-8 -*-
from .ann import *

def get_model(args):
    print('model name:', args.model_name)
    f = globals().get(args.model_name)
    return f(args)
