import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from actor_critic import Actor, Critic
from replaybuffer import ReplayBuffer

class Agent(object):
    def __init__(self, name, args, sess=None, reuse=False, log_tensorboard=False):
        self.name = name
        self._args = args
        self.sess = sess if sess is not None else tf.get_default_session()
        self.reuse = reuse
        self.log_tensorboard = log_tensorboard
        # hyperparameters
        self.gamma = args[name]['gamma']
        self.tau = args[name]['tau']
        self.init_noise_sigma = args[name]['init_noise_sigma']
        self.noise_decay = args[name]['noise_decay']
        # replay buffer
        self.buffer = ReplayBuffer(sample_size=args['batch_size'])
        # env info
        self._setup_env()
        # main actor-critic
        self.actor, self.critic, self.critic_with_actor = self._create_actor_critic()
        # target actor-critic
        self._target_actor, self._target_critic, _ = self._create_actor_critic(is_target=True)

        # loss
        actor_loss, critic_loss = self._loss()
        self.actor.loss += actor_loss
        self.critic.loss += critic_loss
       
        # optimizating operation
        with tf.variable_scope(self.actor.name):
            self.actor.opt_op = self.actor._optimize(self.actor.loss)
        with tf.variable_scope(self.critic.name):
            self.critic.opt_op = self.critic._optimize(self.critic.loss)
        self.opt_op = tf.group(self.actor.opt_op, self.critic.opt_op)

        # target net operations
        target_main_var_pairs = zip(self._target_variables, self.main_variables)
        self.init_target_op = list(map(lambda v: v[0].assign(v[1]), target_main_var_pairs))
        self.update_target_op = list(map(lambda v: v[0].assign(self.tau * v[0] + (1. - self.tau) * v[1]), target_main_var_pairs))

        # operations that add/remove noise from parameters
        self.noise_op, self.denoise_op = self._noise_params()

        self.initialize()

    @property
    def main_variables(self):
        return self.actor.trainable_variables + self.critic.trainable_variables

    @property
    def _target_variables(self):
        return self._target_actor.trainable_variables + self._target_critic.trainable_variables

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_target_op)

    def act(self, state):
        self.sess.run(self.noise_op)
        state = state.reshape((-1, self.state_size))
        action = self.sess.run(self.actor.action, feed_dict={self.actor.state: state})
        self.sess.run(self.denoise_op)
        
        return np.squeeze(action)

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.buffer.sample_size + 100:
            self._learn()

    def restore(self):
        self.actor.restore(self.sess)
        self.critic.restore(self.sess)
        self._target_actor.restore(self.sess)
        self._target_critic.restore(self.sess)

    def save(self):
        self.actor.save(self.sess)
        self.critic.save(self.sess)
        self._target_actor.save(self.sess)
        self._target_critic.save(self.sess)

    def _setup_env(self):
        self.state_size = self._args[self.name]['state_size']
        self.action_size = self._args[self.name]['action_size']
        self.reward_size = self._args[self.name]['reward_size']
        self.env_info = {}
        with tf.name_scope('placeholder'):
            self.env_info['state'] = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
            self.env_info['action'] = tf.placeholder(tf.float32, shape=(None, self.action_size), name='action')
            self.env_info['reward'] = tf.placeholder(tf.float32, shape=(None, self.reward_size), name='reward')
            self.env_info['done'] = tf.placeholder(tf.uint8, shape=(None, 1), name='done')

    def _create_actor_critic(self, is_target=False):
        name_prefix = ''
        if is_target:
            name_prefix += 'target_'

        actor = Actor(name_prefix + 'actor', self._args, self.env_info, self.action_size, reuse=self.reuse, is_target=is_target)
        critic = Critic(name_prefix + 'critic', self._args, self.env_info, reuse=self.reuse, is_target=is_target)
        critic_with_actor = None
        if not is_target:
            critic_with_actor = Critic('critic', self._args, self.env_info, action=actor.action, reuse=True, is_target=is_target)
        return actor, critic, critic_with_actor

    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                actor_loss = - tf.reduce_mean(self.critic_with_actor.Q)

            target_Q = tf.stop_gradient(self.env_info['reward'] + tf.scalar_mul(self.gamma, tf.cast(1 - self.env_info['done'], tf.float32)) * self._target_critic.Q)

            with tf.name_scope('critic_loss'):
                critic_loss = tf.losses.mean_squared_error(target_Q, self.critic.Q)

        return actor_loss, critic_loss

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()

        # update actor
        self.sess.run(self.actor.opt_op, feed_dict={
            self.env_info['state']: states
        })

        # update critic
        next_target_action = self.sess.run(self._target_actor.action, feed_dict={
            self._target_actor.state: next_states,
        })
        next_target_Q = self.sess.run(self._target_critic.Q, feed_dict={
            self._target_critic.state: next_states,
            self._target_critic.action: next_target_action,
        })
        self.sess.run(self.critic.opt_op, feed_dict={
            self.env_info['reward']: rewards,
            self.env_info['done']: dones,
            self._target_critic.Q: next_target_Q,
            self.critic.state: states,
            self.critic.action: actions,
        })

        # update the target networks
        self.sess.run(self.update_target_op)

    def _noise_params(self):
        # add noise to parameters
        # TODO: only to perturbable params
        noises = []
        noise_sigma = tf.get_variable('noise_sigma', initializer=self.init_noise_sigma, 
                                      trainable=False, regularizer=tc.layers.l2_regularizer(1 - self.noise_decay))
        if self.log_tensorboard:
            tf.summary.scalar('noise_sigma', noise_sigma)
        for var in self.actor.trainable_variables:
            noise = tf.truncated_normal(tf.shape(var), stddev=noise_sigma)
            noises.append(noise)
        
        param_noise_pairs = zip(self.actor.trainable_variables, noises)
        noise_op = list(map(lambda v: v[0].assign(v[0] + v[1]), param_noise_pairs))
        denoise_op = list(map(lambda v: v[0].assign(v[0] - v[1]), param_noise_pairs))

        return noise_op, denoise_op
