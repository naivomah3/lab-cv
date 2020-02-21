import math
import os
import random
import string

def get_rand_name(size=30, chars=string.ascii_letters + string.digits):
    '''
    Generate Random filename
    rtype: a random string of length 30
    '''
    return ''.join(random.choice(chars) for x in range(size))

def mk_dir(*args):
    dir = os.path.join(os.getcwd(), *args)
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
        return True
    return False

def lr_decay(epoch):
    '''
    Learning rate scheduler
    '''
    init_lr = 0.01
    drop = 0.7
    frequency_drop = 5.0
    lr = init_lr * math.pow(drop, math.floor((1 + epoch) / frequency_drop))
    return lr


