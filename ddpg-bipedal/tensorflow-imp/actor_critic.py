import tensorflow as tf
import tensorflow.contrib as tc
from module import SubModule
import utils.tf_utils as tf_utils

"""
 These classes are designed to cooperate with DDPG,
 which means they are not supposed to be used independently
"""
class Actor(SubModule):
    def __init__(self, name, args, env_info, action_size, reuse=False, build_graph=True, log_tensorboard=True, is_target=False):
        self.action_size = action_size
        self.state = env_info['state'] if not is_target else env_info['next_state']
        self.is_target = is_target
        self.variable_scope = 'ddpg/' + ('target/' if self.is_target else 'main/') + name
        super(Actor, self).__init__(name, args, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)

    @property
    def global_variables(self):
        return tf.global_variables(scope=self.variable_scope)

    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self.variable_scope)

    def _build_graph(self):
        self.action = self._network(self.state)

        if not self.is_target:
            # following operations will be further defined in DDPG
            self.loss = None
            self.opt_op = None
    
    def _network(self, state):
        x = self._dense_norm_activation(state, 512, kernel_initializer=tf_utils.kaiming_initializer(), activation=tf.nn.relu)
        x = self._dense_norm_activation(x, 256, kernel_initializer=tf_utils.kaiming_initializer(), activation=tf.nn.relu)
        x = self._dense_norm_activation(x, self.action_size, activation=tf.tanh)

        return x


class Critic(SubModule):
    def __init__(self, name, args, env_info, action=None, reuse=False, build_graph=True, log_tensorboard=True, is_target=False):
        self.state = env_info['state'] if not is_target else env_info['next_state']
        self.action = env_info['action'] if action is None else action
        self.is_target = is_target
        self.variable_scope = 'ddpg/' + ('target/' if self.is_target else 'main/') + name
        super(Critic, self).__init__(name, args, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)

    @property
    def global_variables(self):
        return tf.global_variables(scope=self.variable_scope)

    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self.variable_scope)
        
    def _build_graph(self):
        self.Q = self._network(self.state, self.action)
        
        if not self.is_target:
            # following operations will be further defined in DDPG
            self.loss = None
            self.opt_op = None

    def _network(self, state, action):
        x = self._dense_norm_activation(state, 512, kernel_initializer=tf_utils.kaiming_initializer(), activation=tf.nn.relu)
        x = tf.concat([x, action], 1)
        x = self._dense_norm_activation(x, 256, kernel_initializer=tf_utils.kaiming_initializer(), activation=tf.nn.relu)
        x = self._dense(x, 1)

        return x

