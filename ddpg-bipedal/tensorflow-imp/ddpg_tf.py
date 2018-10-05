import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from utils import tf_utils
from module import Model
from actor_critic import ActorCritic
from replaybuffer import ReplayBuffer

class DDPG(Model):
    """ Interface """
    def __init__(self, name, args, sess=None, reuse=False, log_tensorboard=True, save=True):
        self.learn_steps = 0

        # hyperparameters
        self.gamma = args[name]['gamma']
        self.tau = args[name]['tau']
        self.init_noise_sigma = args[name]['init_noise_sigma']
        self.noise_decay = args[name]['noise_decay']
        
        # replay buffer
        self.buffer = ReplayBuffer(sample_size=args['batch_size'], max_len=args[name]['buffer_size'])
        
        super(DDPG, self).__init__(name, args, sess=sess, reuse=reuse, build_graph=True, log_tensorboard=log_tensorboard, save=save)

        self._initialize_target_net()

    @property
    def main_variables(self):
        return self.actor_critic.trainable_variables

    @property
    def _target_variables(self):
        return self._target_actor_critic.trainable_variables

    def act(self, state):
        self.sess.run(self.noise_op)
        state = state.reshape((-1, self.state_size))
        action = self.sess.run(self.actor_critic.actor_action, feed_dict={self.actor_critic.state: state})
        
        return np.squeeze(action)

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.buffer.sample_size + 100:
            self._learn()

    """ Implementation """
    def _build_graph(self):
        # env info
        self._setup_env()
        
        # main actor-critic
        self.actor_critic = self._create_actor_critic()
        # target actor-critic
        self._target_actor_critic = self._create_actor_critic(is_target=True)

        # losses
        self.actor_loss, self.critic_loss = self._loss()
    
        # optimizating operation
        self.opt_op = self._optimize([self.actor_loss, self.critic_loss])

        # target net update operations
        self.init_target_op, self.update_target_op = self._targetnet_ops() 
        
        # operations that add/remove noise from parameters
        self.noise_op = self._noise_params()
        
    def _setup_env(self):
        self.state_size = self._args[self.name]['state_size']
        self.action_size = self._args[self.name]['action_size']
        self.env_info = {}
        with tf.name_scope('placeholders'):
            self.env_info['state'] = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
            self.env_info['action'] = tf.placeholder(tf.float32, shape=(None, self.action_size), name='action')
            self.env_info['next_state'] = tf.placeholder(tf.float32, shape=(None, self.state_size), name='next_state')
            self.env_info['reward'] = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
            self.env_info['done'] = tf.placeholder(tf.uint8, shape=(None, 1), name='done')

    def _create_actor_critic(self, is_target=False):
        name = 'target_actor_critic' if is_target else 'actor_critic'
        log_tensorboard = False if is_target else True
        actor_critic = ActorCritic(name, self._args, self.env_info, self.action_size, reuse=self.reuse, log_tensorboard=log_tensorboard, is_target=is_target)
        
        return actor_critic

    def _loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('l2_loss'):
                encoder_l2_loss = tf.losses.get_regularization_loss(scope='ddpg/actor_critic/encoder', name='encoder_l2_loss')
                actor_l2_loss = tf.losses.get_regularization_loss(scope='ddpg/actor_critic/actor', name='actor_l2_loss')
                critic_l2_loss = tf.losses.get_regularization_loss(scope='ddpg/actor_critic/critic', name='critic_l2_loss')

            with tf.name_scope('actor_loss'):
                actor_loss = tf.negative(tf.reduce_mean(self.actor_critic.Q_with_actor), name='actor_loss') + encoder_l2_loss + actor_l2_loss

            with tf.name_scope('critic_loss'):
                target_Q = tf.stop_gradient(self.env_info['reward'] 
                                            + self.gamma * tf.cast(1 - self.env_info['done'], tf.float32) * self._target_actor_critic.Q_with_actor, name='target_Q')
                critic_loss = tf.losses.mean_squared_error(target_Q, self.actor_critic.Q) + encoder_l2_loss + critic_l2_loss
            
            if self.log_tensorboard:
                tf.summary.scalar('actor_l2_loss_', actor_l2_loss)
                tf.summary.scalar('critic_l2_loss_', critic_l2_loss)
                tf.summary.scalar('encoder_l2_loss_', encoder_l2_loss)
                tf.summary.scalar('actor_loss_', actor_loss)
                tf.summary.scalar('critic_loss_', critic_loss)

        return actor_loss, critic_loss

    def _optimize(self, losses):
        with tf.variable_scope('optimizer'):
            actor_loss, critic_loss = losses
            actor_opt_op = self._optimize_objective(actor_loss, 'actor')
            critic_opt_op = self._optimize_objective(critic_loss, 'critic')

            opt_op = tf.group(actor_opt_op, critic_opt_op)

        return opt_op

    def _optimize_objective(self, loss, name):
        # params for optimizer
        learning_rate = self._args['actor_critic'][name]['learning_rate'] if 'learning_rate' in self._args['actor_critic'][name] else 1e-3
        beta1 = self._args['actor_critic'][name]['beta1'] if 'beta1' in self._args['actor_critic'][name] else 0.9
        beta2 = self._args['actor_critic'][name]['beta2'] if 'beta2' in self._args['actor_critic'][name] else 0.999
        clip_norm = self._args[name]['actor_critic']['clip_norm'] if 'clip_norm' in self._args['actor_critic'] else 5.

        with tf.variable_scope(name+'_opt', reuse=self.reuse):
            # setup optimizer
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

            tvars = self.actor_critic.actor_trainable_variables if name == 'actor' else self.actor_critic.critic_trainable_variables
            grads, tvars = list(zip(*self._optimizer.compute_gradients(loss, var_list=tvars)))
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)
            opt_op = self._optimizer.apply_gradients(zip(grads, tvars))

        if self.log_tensorboard:
            with tf.name_scope(name):
                with tf.name_scope('gradients_'):
                    for grad, var in zip(grads, tvars):
                        if grad is not None:
                            tf.summary.histogram(var.name.replace(':0', ''), grad)
                with tf.name_scope('params_'):
                    for var in tvars:
                        tf.summary.histogram(var.name.replace(':0', ''), var)

        return opt_op

    def _targetnet_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self._target_variables, self.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], self.tau * v[1] + (1. - self.tau) * v[0], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op

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
            self.writer.add_summary(summary, self.learn_steps)
        else:
            _ = self.sess.run(self.opt_op, feed_dict=feed_dict)

        # update the target networks
        self.sess.run(self.update_target_op)

        self.learn_steps += 1

    def _noise_params(self):
        with tf.variable_scope('noise'):
            noise_sigma = tf.get_variable('noise_sigma', initializer=self.init_noise_sigma, 
                                          trainable=False)

            noise_decay_op = tf.assign(noise_sigma, self.noise_decay * noise_sigma, name='noise_decay_op')

            noises = []
            for var in self.actor_critic.actor_perturbable_variables:
                noise = tf.truncated_normal(tf.shape(var), stddev=noise_sigma)
                noises.append(noise)
            
            if self.log_tensorboard:
                tf.summary.scalar('noise_sigma_', noise_sigma)

            param_noise_pairs = zip(self.actor_critic.actor_perturbable_variables, noises)

            with tf.control_dependencies([noise_decay_op]):
                noise_op = list(map(lambda v: tf.assign(v[0], v[0] + v[1], name='noise_op'), param_noise_pairs))

        return noise_op

    def _initialize_target_net(self):
        self.sess.run(self.init_target_op)
