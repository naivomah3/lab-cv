import math
import os
import random
import string
import pandas as pd
from keras.activations import relu
from keras import backend as K
import tensorflow as tf

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


def get_flops(file_path, model_name):
    if not os.path.exists(file_path):
        df = pd.DataFrame({'model': [],
                           'flops':[],
                           'params': []})
        df.to_csv(file_path)

    run_meta = tf.RunMetadata()
    sess = K.get_session()
    K.set_session(sess)

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    # total_flops = int("{:,}".format(flops.total_float_ops))
    # total_params = int("{:,}".format(params.total_parameters))

    # Append DF
    data = {
        'model': [model_name],
        'flops': [f"{flops.total_float_ops/1e6:.3}"],
        'params': [f"{params.total_parameters/1e6:.3}"]
    }
    df = pd.DataFrame(data, columns=['model', 'flops', 'params'])

    df.to_csv(file_path, mode='a', header=False)

    return flops.total_float_ops, params.total_parameters



