import tensorflow as tf
import tensorflow.contrib as tc

# kaiming initializer
def kaiming_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tc.layers.variance_scaling_initializer(factor=2., mode='FAN_IN', uniform=uniform, seed=seed, dtype=dtype)

# xavier initializer
def xavier_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tc.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=uniform, seed=seed, dtype=dtype)

# batch normalization and relu
def bn_relu(layer, training): 
    return tf.nn.relu(tf.layers.batch_normalization(layer, training=training))

# layer normalization and relu
def ln_relu(layer):
    return tf.nn.relu(tc.layers.layer_norm(layer))

def standard_normalization(images):
    mean, var = tf.nn.moments(images, [0, 1, 2])
    std = tf.sqrt(var)

    normalized_images = (images - mean) / std
    
    return normalized_images, mean, std

def range_normalization(images, normalizing=True):
    if normalizing:
        processed_images = tf.cast(images, tf.float32) / 128 - 1
    else:
        processed_images = tf.cast((tf.clip_by_value(images, -1, 1) + 1) * 128, tf.uint8)

    return processed_images

def logsumexp(value, axis=None, keepdims=False):
    if axis is not None:
        max_value = tf.reduce_max(value, axis=axis, keepdims=True)
        value0 = value - max_value    # for numerical stability
        if keepdims is False:
            max_value = tf.squeeze(max_value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value0),
                                                axis=axis, keepdims=keepdims))
    else:
        max_value = tf.reduce_max(value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value - max_value)))