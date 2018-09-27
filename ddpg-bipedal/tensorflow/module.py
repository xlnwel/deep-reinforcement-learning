import tensorflow as tf
import tensorflow.contrib as tc
import utils.utils as utils
import utils.tf_utils as tf_utils
import os
import sys

class Module(object):
    """ Interface """
    def __init__(self, name, args, reuse=False, build_graph=True, log_tensorboard=False, save=True):
        self._args = args
        self.name = name
        self.reuse = reuse
        self.log_tensorboard = log_tensorboard

        if build_graph:
            self.build_graph(save)

    @property
    def global_variables(self):
        return tf.global_variables(scope=self.name)

    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self.name)
        
    @property
    def perturbable_variables(self):
        return [var for var in self.trainable_variables if 'LayerNorm' not in var.name]

    def build_graph(self, save=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            scale = self._args[self.name]['weight_decay'] if self.name in self._args and 'weight_decay' in self._args[self.name] else 0.
            self.l2_regularizer = tc.layers.l2_regularizer(scale)

            self._build_graph()

            if save:
                self._saver = tf.train.Saver(self.global_variables)
            else:
                self._saver = None
    
    def restore(self, filename=None):
        """ To restore the most recent model, simply leave filename None
        To restore a specific version of model, set filename to the model stored in saved_models
        """
        if self._saver:
            NO_SUCH_FILE = 'Missing_file'
            if filename:
                path_prefix = os.path.join(sys.path[0], 'saved_models/' + filename, self.name)
            else:
                models = self._get_models()
                key = self._get_model_name()
                path_prefix = models[key] if key in models else NO_SUCH_FILE
            if path_prefix != NO_SUCH_FILE:
                try:
                    self._saver.restore(self.sess, path_prefix)
                    print("Params for {} are restored.".format(self.name))
                    return 
                except:
                    del models[key]
            print('No saved model for "{}" is found. \nStart Training from Scratch!'.format(self.name))

    def save(self):
        if self._saver:
            key = self._get_model_name()
            path_prefix = self._saver.save(self.sess, os.path.join(sys.path[0], 'saved_models/' + self._args['model_name'], str(self.name)))
            # save the model name to models.yaml
            utils.save_args({key: path_prefix}, filename='models.yaml')

    """ Implementation """
    def _build_graph(self):
        raise NotImplementedError

    def _optimize(self, loss):
        # params for optimizer
        init_learning_rate = self._args[self.name]['learning_rate'] if 'learning_rate' in self._args[self.name] else 1e-3
        beta1 = self._args[self.name]['beta1'] if 'beta1' in self._args[self.name] else 0.9
        beta2 = self._args[self.name]['beta2'] if 'beta2' in self._args[self.name] else 0.999
        decay_rate = self._args[self.name]['decay_rate'] if 'decay_rate' in self._args[self.name] else 1
        decay_steps = self._args[self.name]['decay_steps'] if 'decay_steps' in self._args[self.name] else 100000

        with tf.variable_scope('optimizer', reuse=self.reuse):
            # setup optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer([0]), trainable=False)
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
            
            if self.log_tensorboard:
                tf.summary.scalar('learning_rate_', learning_rate)

            with tf.control_dependencies(update_ops):
                tvars = self.trainable_variables
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
                opt_op = self._optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
                if self.log_tensorboard:
                    with tf.name_scope('gradients_'):
                        for grad, var in zip(grads, tvars):
                            if grad is not None:
                                tf.summary.histogram(var.name.replace(':0', ''), grad)

        if self.log_tensorboard:
            with tf.name_scope('weights_'):
                for var in self.trainable_variables:
                    tf.summary.histogram(var.name.replace(':0', ''), var)
            
        return opt_op

    def _dense(self, x, units, kernel_initializer=tf_utils.xavier_initializer()):
        return tf.layers.dense(x, units, kernel_initializer=kernel_initializer, 
                               kernel_regularizer=self.l2_regularizer)

    def _dense_bn_relu(self, x, units, kernel_initializer=tf_utils.kaiming_initializer()):
        x = self._dense(x, units, kernel_initializer=kernel_initializer)
        x = tf_utils.bn_relu(x, self._training())

        return x

    def _dense_ln_relu(self, x, units, kernel_initializer=tf_utils.kaiming_initializer()):
        x = self._dense(x, units, kernel_initializer=kernel_initializer)
        x = tf_utils.ln_relu(x)

        return x

    def _dense_norm_activation(self, x, units, kernel_initializer=tf_utils.xavier_initializer(),
                               normalization=None, activation=None):
        x = self._dense(x, units, kernel_initializer=kernel_initializer)
        x = tf_utils.norm_activation(x, normalization=normalization, activation=activation, training=self._training())

        return x

    def _conv(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=tf_utils.xavier_initializer()): 
        return tf.layers.conv2d(x, filters, filter_size, 
                                strides=strides, padding=padding, 
                                kernel_initializer=kernel_initializer, 
                                kernel_regularizer=self.l2_regularizer)

    def _conv_bn_relu(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=tf_utils.kaiming_initializer()):
        x = self._conv(x, filters, filter_size, strides, padding=padding, kernel_initializer=kernel_initializer)
        x = tf_utils.bn_relu(x, self._training())

        return x

    def _conv_norm_activation(self, x, filters, filter_size, strides=1, padding='same', 
                              kernel_initializer=tf_utils.xavier_initializer(), normalization=None, activation=None):
        x = self._conv(x, filters, filter_size, strides, padding=padding, kernel_initializer=kernel_initializer)
        x = tf_utils.norm_activation(x, normalization=normalization, activation=activation, training=getattr(self, 'is_training', False))

        return x
    
    def _convtrans(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=tf_utils.xavier_initializer()): 
        return tf.layers.conv2d_transpose(x, filters, filter_size, strides=strides, padding=padding, 
                                          kernel_initializer=kernel_initializer, kernel_regularizer=self.l2_regularizer)
    
    def _convtrans_bn_relu(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=tf_utils.kaiming_initializer()):
        x = self._convtrans(x, filters, filter_size, strides, padding=padding)
        x = tf_utils.bn_relu(x, self._training())

        return x

    def _convtrans_norm_activation(self, x, filters, filter_size, strides=1, padding='same', 
                              kernel_initializer=tf_utils.xavier_initializer(), normalization=None, activation=None):
        x = self._convtrans(x, filters, filter_size, strides, padding=padding, kernel_initializer=kernel_initializer)
        x = tf_utils.norm_activation(x, normalization=normalization, activation=activation, training=getattr(self, 'is_training', False))

        return x

    def _conv_pool_bn_relu(self, x, filters, filter_size, strides=1):
        y = x
        y = self._conv(y, filters, filter_size, strides, padding='same', kernel_initializer=tf_utils.kaiming_initializer())
        y = tf_utils.bn_relu(y, self._training())
        x = tf.layers.average_pooling2d(x, strides, strides, padding='same')

        return tf.concat([x, y], -1)

    def _training(self):
        return getattr(self, 'is_training', False)
        
    def _get_models(self):
        return utils.load_args('models.yaml')

    def _get_model_name(self):
        return self.name + '_' + self._args['model_name']
