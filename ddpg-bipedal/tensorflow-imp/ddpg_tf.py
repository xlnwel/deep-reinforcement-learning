import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from utils import tf_utils
from module import Module
from actor_critic import Actor, Critic
from replaybuffer import ReplayBuffer

class Agent(Module):
    def __init__(self, name, args, sess=None, reuse=False, log_tensorboard=True, save=True):
        self.name = name
        self._args = args
        self.sess = sess if sess is not None else tf.get_default_session()
        self.reuse = reuse
        self.log_tensorboard = log_tensorboard
        self.train_steps = 0

        # hyperparameters
        self.gamma = args[name]['gamma']
        self.tau = args[name]['tau']
        self.init_noise_sigma = args[name]['init_noise_sigma']
        self.noise_decay = args[name]['noise_decay']
        
        # replay buffer
        self.buffer = ReplayBuffer(sample_size=args['batch_size'], max_len=args[name]['buffer_size'])

        super(Agent, self).__init__(name, args, reuse=reuse, build_graph=True, log_tensorboard=log_tensorboard, save=save)

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

    def _build_graph(self):
        # env info
        self._setup_env()
        
        # main actor-critic
        self.actor, self.critic, self.critic_with_actor = self._create_actor_critic()
        # target actor-critic
        self._target_actor, self._target_critic, self._target_critic_with_actor = self._create_actor_critic(is_target=True)

        # losses
        actor_loss, critic_loss = self._loss()
        self.actor.loss = actor_loss
        self.critic.loss = critic_loss
    
        # optimizating operation
        with tf.variable_scope('optimization'):
            with tf.variable_scope('actor'):
                self.actor.opt_op = self.actor._optimize(self.actor.loss)
            with tf.variable_scope('critic'):
                self.critic.opt_op = self.critic._optimize(self.critic.loss)
        self.opt_op = tf.group(self.actor.opt_op, self.critic.opt_op)

        if self.log_tensorboard:
            tf.summary.scalar('Q_', tf.reduce_mean(self.critic.Q))
            
        # target net update operations
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self._target_variables, self.main_variables))
            self.init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            self.update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))
            
        # operations that add/remove noise from parameters
        self.noise_op, self.denoise_op = self._noise_params()
        
        # tensorboard info
        self._setup_tensorboard_summary()
        
    def _setup_env(self):
        self.state_size = self._args[self.name]['state_size']
        self.action_size = self._args[self.name]['action_size']
        self.env_info = {}
        with tf.name_scope('placeholder'):
            self.env_info['state'] = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
            self.env_info['action'] = tf.placeholder(tf.float32, shape=(None, self.action_size), name='action')
            self.env_info['next_state'] = tf.placeholder(tf.float32, shape=(None, self.state_size), name='next_state')
            self.env_info['reward'] = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
            self.env_info['done'] = tf.placeholder(tf.uint8, shape=(None, 1), name='done')

    def _setup_tensorboard_summary(self):
        if self.log_tensorboard:
            self.writer = tf.summary.FileWriter(os.path.join('./tensorboard', self._args['model_name']), self.sess.graph)
            self.merged_op = tf.summary.merge_all()

    def _create_actor_critic(self, is_target=False):
        with tf.variable_scope('target' if is_target else 'main'):
            actor = Actor('actor', self._args, self.env_info, self.action_size, reuse=self.reuse, is_target=is_target, save=False)
            critic = Critic('critic', self._args, self.env_info, reuse=self.reuse, is_target=is_target, save=False)
            critic_with_actor = Critic('critic', self._args, self.env_info, action=actor.action, reuse=True, is_target=is_target, save=False)
        
        return actor, critic, critic_with_actor

    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('actor_loss'):
                # actor_l2_loss = tf.losses.get_regularization_loss(scope='ddpg/main/actor')
                actor_loss = - tf.reduce_mean(self.critic_with_actor.Q)# + actor_l2_loss

            with tf.name_scope('critic_loss'):
                target_Q = tf.stop_gradient(self.env_info['reward'] 
                                            + self.gamma * tf.cast(1 - self.env_info['done'], tf.float32) * self._target_critic_with_actor.Q)
                # critic_l2_loss = tf.losses.get_regularization_loss(scope='ddpg/main/critic')
                critic_loss = tf.losses.mean_squared_error(target_Q, self.critic.Q)# + critic_l2_loss

            if self.log_tensorboard:
                # tf.summary.scalar('actor_l2_loss_', actor_l2_loss)
                # tf.summary.scalar('critic_l2_loss_', critic_l2_loss)
                tf.summary.scalar('actor_loss_', actor_loss)
                tf.summary.scalar('critic_loss_', critic_loss)
                with tf.variable_scope('debug_grads'):
                    tensors = [tf_utils.get_tensor(self.sess, op_name='ddpg/main/actor/Tanh'), tf_utils.get_tensor(self.sess, op_name='ddpg/main/actor/dense_2/BiasAdd')]
                    tvars = tensors + self.actor.trainable_variables
                    grads = tf.gradients(actor_loss, tvars)
                    for grad, var in zip(grads, tvars):
                        tf.summary.histogram(var.name.replace(':0', ''), grad)


        return actor_loss, critic_loss

    def _learn(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()

        feed_dict = {
            self.env_info['state']: states,
            self.env_info['action']: actions,
            self.env_info['reward']: rewards,
            self.env_info['next_state']: next_states,
            self.env_info['done']: dones,
        }

        # update the main networks
        if self.log_tensorboard:
            _, summary = self.sess.run([self.opt_op, self.merged_op], feed_dict=feed_dict)
            self.writer.add_summary(summary, self.train_steps)
        else:
            _ = self.sess.run(self.opt_op, feed_dict=feed_dict)

        # update the target networks
        self.sess.run(self.update_target_op)

        self.train_steps += 1

    def _noise_params(self):
        with tf.variable_scope('noise'):
            noise_sigma = tf.get_variable('noise_sigma', initializer=self.init_noise_sigma, 
                                          trainable=False)

            noise_decay_op = tf.assign(noise_sigma, self.noise_decay * noise_sigma, name='noise_decay_op')

            noises = []
            for var in self.actor.perturbable_variables:
                noise = tf.truncated_normal(tf.shape(var), stddev=noise_sigma)
                noises.append(noise)
            
            if self.log_tensorboard:
                tf.summary.scalar('noise_sigma_', noise_sigma)

            param_noise_pairs = zip(self.actor.perturbable_variables, noises)

            with tf.control_dependencies([noise_decay_op]):
                noise_op = list(map(lambda v: tf.assign(v[0], v[0] + v[1], name='noise_op'), param_noise_pairs))
                denoise_op = list(map(lambda v: tf.assign(v[0], v[0] - v[1], name='denoise_op'), param_noise_pairs))

        return noise_op, denoise_op
