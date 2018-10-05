import tensorflow as tf
import tensorflow.contrib as tc
from module import Module
import utils.tf_utils as tf_utils

class ActorCritic(Module):
    def __init__(self, name, args, env_info, action_size, reuse=False, log_tensorboard=True, is_target=False):
        self.action_size = action_size
        self.state = env_info['state'] if not is_target else env_info['next_state']
        self.action = env_info['action']
        self.is_target = is_target
        self.variable_scope = 'ddpg/' + name
        super(ActorCritic, self).__init__(name, args, reuse=reuse, log_tensorboard=log_tensorboard)

    @property
    def state_encoder_variables(self):
        return [] if self.is_target else tf.trainable_variables(scope=self.variable_scope + '/state_encoder')

    @property
    def actor_trainable_variables(self):
        return [] if self.is_target else tf.trainable_variables(scope=self.variable_scope + '/actor') + self.state_encoder_variables

    @property
    def critic_trainable_variables(self):
        return [] if self.is_target else tf.trainable_variables(scope=self.variable_scope + '/critic') + self.state_encoder_variables

    @property
    def actor_perturbable_variables(self):
        return [var for var in self.actor_trainable_variables if 'LayerNorm' not in var.name]

    @property
    def trainable_variables(self):
        return self.actor_trainable_variables + self.critic_trainable_variables

    def _build_graph(self):
        self.actor_action = self._actor(self.state)
        self.Q = self._critic(self.state, self.action, reuse=self.reuse)
        self.Q_with_actor = self._critic(self.state, self.actor_action, reuse=True)
        
        if self.log_tensorboard:
            tf.summary.scalar('Q_', tf.reduce_mean(self.Q))
            tf.summary.scalar('Q_with_actor_', tf.reduce_mean(self.Q_with_actor))

    def _encode_state(self, state, reuse):
        with tf.variable_scope('state_encoder', reuse=reuse):
            x = self._dense_norm_activation(state, 512, kernel_initializer=tf_utils.kaiming_initializer(), activation=tf.nn.relu)

        return x

    def _actor(self, state):
        x = self._encode_state(state, self.reuse)
        with tf.variable_scope('actor', reuse=self.reuse):
            x = self._dense_norm_activation(x, 256, kernel_initializer=tf_utils.kaiming_initializer(), activation=tf.nn.relu)
            x = self._dense_norm_activation(x, self.action_size)
            x = tf.clip_by_value(x, -1, 1, name='action')

        return x

    def _critic(self, state, action, reuse):
        x = self._encode_state(state, True)
        with tf.variable_scope('critic', reuse=reuse):
            x = tf.concat([x, action], 1)
            x = self._dense_norm_activation(x, 256, kernel_initializer=tf_utils.kaiming_initializer(), activation=tf.nn.relu)
            x = self._dense(x, 1)

        return x