import math
import os
import random
import string
from keras.activations import relu

def get_rand_name(size=30, chars=string.ascii_letters + string.digits):
    '''
    Generate Random filename
    rtype: a random string of length 30
    '''
    return ''.join(random.choice(chars) for x in range(size))

def mk_dir(*args):
    dir = os.path.join(os.getcwd(), *args)
    # print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
        return True
    return False

def lr_decay(epoch):
    '''
    Learning rate scheduler
    '''
    init_lr = 0.01
    drop_factor = 0.7
    drop_freq = 5.0
    lr = init_lr * math.pow(drop_factor, math.floor((1 + epoch) / drop_freq))
    return lr


# DeepLab helpers
def relu6(layer):
    return relu(layer, max_value=6)

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


