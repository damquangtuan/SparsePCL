# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Objectives to compute loss and value targets.

Implements Actor Critic, PCL (vanilla PCL, Unified PCL, Trust PCL), and TRPO.
"""

import tensorflow as tf
import numpy as np
import sys

class Objective(object):
  def __init__(self, learning_rate, clip_norm):
    self.learning_rate = learning_rate
    self.clip_norm = clip_norm

  def get_optimizer(self, learning_rate):
    """Optimizer for gradient descent ops."""
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=2e-4)

  def training_ops(self, loss, learning_rate=None):
    """Gradient ops."""
    opt = self.get_optimizer(learning_rate)
    params = tf.trainable_variables()
    grads = tf.gradients(loss, params)

    if self.clip_norm:
      grads, global_norm = tf.clip_by_global_norm(grads, self.clip_norm)
      tf.summary.scalar('grad_global_norm', global_norm)

    return opt.apply_gradients(list(zip(grads, params)))

  def get(self, rewards, pads, values, final_values,
          log_probs, prev_log_probs, target_log_probs,
          entropies, logits,
          target_values, final_target_values, actions=None):
    """Get objective calculations."""
    raise NotImplementedError()


def discounted_future_sum(values, discount, rollout):
  """Discounted future sum of time-major values."""
  discount_filter = tf.reshape(
      discount ** tf.range(float(rollout)), [-1, 1, 1])
  expanded_values = tf.concat(
      [values, tf.zeros([rollout - 1, tf.shape(values)[1]])], 0)

  conv_values = tf.transpose(tf.squeeze(tf.nn.conv1d(
      tf.expand_dims(tf.transpose(expanded_values), -1), discount_filter,
      stride=1, padding='VALID'), -1))

  return conv_values


def discounted_two_sided_sum(values, discount, rollout):
  """Discounted two-sided sum of time-major values."""
  roll = float(rollout)
  discount_filter = tf.reshape(
      discount ** tf.abs(tf.range(-roll + 1, roll)), [-1, 1, 1])
  expanded_values = tf.concat(
      [tf.zeros([rollout - 1, tf.shape(values)[1]]), values,
       tf.zeros([rollout - 1, tf.shape(values)[1]])], 0)

  conv_values = tf.transpose(tf.squeeze(tf.nn.conv1d(
      tf.expand_dims(tf.transpose(expanded_values), -1), discount_filter,
      stride=1, padding='VALID'), -1))

  return conv_values


def shift_values(values, discount, rollout, final_values=0.0):
  """Shift values up by some amount of time.

  Those values that shift from a value beyond the last value
  are calculated using final_values.

  """
  roll_range = tf.cumsum(tf.ones_like(values[:rollout, :]), 0,
                         exclusive=True, reverse=True)
  final_pad = tf.expand_dims(final_values, 0) * discount ** roll_range
  return tf.concat([discount ** rollout * values[rollout:, :],
                    final_pad], 0)


def spmax_tau(logits):
    batch_size = tf.shape(logits)[0]
    num_actions = tf.shape(logits)[1]

    z = logits

    z_sorted, _ = tf.nn.top_k(z, k=num_actions)

    z_cumsum = tf.cumsum(z_sorted, axis=1)
    k = tf.range(1, tf.cast(num_actions, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum

    k_z = tf.reduce_sum(tf.cast(z_check, tf.int32), axis=1)

    indices = tf.stack([tf.range(0, batch_size), k_z - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    return tau_z

class ActorCritic(Objective):
  """Standard Actor-Critic."""

  def __init__(self, learning_rate, clip_norm=5,
               policy_weight=1.0, critic_weight=0.1,
               tau=0.1, gamma=1.0, rollout=10,
               eps_lambda=0.0, clip_adv=None,
               use_target_values=False,q=2.0,k=0.5):
    super(ActorCritic, self).__init__(learning_rate, clip_norm=clip_norm)
    self.policy_weight = policy_weight
    self.critic_weight = critic_weight
    self.tau = tau
    self.gamma = gamma
    self.rollout = rollout
    self.clip_adv = clip_adv

    self.eps_lambda = tf.get_variable(  # TODO: need a better way
        'eps_lambda', [], initializer=tf.constant_initializer(eps_lambda),
        trainable=False)
    self.new_eps_lambda = tf.placeholder(tf.float32, [])
    self.assign_eps_lambda = self.eps_lambda.assign(
        0.99 * self.eps_lambda + 0.01 * self.new_eps_lambda)
    self.use_target_values = use_target_values
    self.q = q
    self.k = k

  def get(self, rewards, pads, values, final_values,
          log_probs, prev_log_probs, target_log_probs,
          entropies, logits,
          target_values, final_target_values, actions=None):
    not_pad = 1 - pads
    batch_size = tf.shape(rewards)[1]

    entropy = not_pad * sum(entropies)
    rewards = not_pad * rewards
    value_estimates = not_pad * values[:, :, 0]
    log_probs = not_pad * sum(log_probs)
    target_values = not_pad * tf.stop_gradient(target_values)
    final_target_values = tf.stop_gradient(final_target_values)

    sum_rewards = discounted_future_sum(rewards, self.gamma, self.rollout)
    if self.use_target_values:
      last_values = shift_values(
          target_values, self.gamma, self.rollout,
          final_target_values)
    else:
      last_values = shift_values(value_estimates, self.gamma, self.rollout,
                                 final_values)

    future_values = sum_rewards + last_values
    baseline_values = value_estimates

    adv = tf.stop_gradient(-baseline_values + future_values)
    if self.clip_adv:
      adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))
    policy_loss = -adv * log_probs
    critic_loss = -adv * baseline_values
    regularizer = -self.tau * entropy

    policy_loss = tf.reduce_mean(
        tf.reduce_sum(policy_loss * not_pad, 0))
    critic_loss = tf.reduce_mean(
        tf.reduce_sum(critic_loss * not_pad, 0))
    regularizer = tf.reduce_mean(
        tf.reduce_sum(regularizer * not_pad, 0))

    # loss for gradient calculation
    loss = (self.policy_weight * policy_loss +
            self.critic_weight * critic_loss + regularizer)

    raw_loss = tf.reduce_mean(  # TODO
        tf.reduce_sum(not_pad * policy_loss, 0))

    gradient_ops = self.training_ops(
        loss, learning_rate=self.learning_rate)

    tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0))
    tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0))
    tf.summary.scalar('avg_rewards',
                      tf.reduce_mean(tf.reduce_sum(rewards, 0)))
    tf.summary.scalar('policy_loss',
                      tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
    tf.summary.scalar('critic_loss',
                      tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('raw_loss', raw_loss)

    return (loss, raw_loss, future_values,
            gradient_ops, tf.summary.merge_all())


class PCL(ActorCritic):
  """PCL implementation.

  Implements vanilla PCL, Unified PCL, and Trust PCL depending
  on provided inputs.

  """

  def get(self, rewards, pads, values, final_values,
          log_probs, prev_log_probs, target_log_probs,
          entropies, logits,
          target_values, final_target_values, actions=None):
    not_pad = 1 - pads
    batch_size = tf.shape(rewards)[1]

    rewards = not_pad * rewards
    value_estimates = not_pad * values[:, :, 0]
    log_probs = not_pad * sum(log_probs)
    target_log_probs = not_pad * tf.stop_gradient(sum(target_log_probs))
    relative_log_probs = not_pad * (log_probs - target_log_probs)
    target_values = not_pad * tf.stop_gradient(target_values)
    final_target_values = tf.stop_gradient(final_target_values)

    # Prepend.
    not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                         not_pad], 0)
    rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                         rewards], 0)
    value_estimates = tf.concat(
        [self.gamma ** tf.expand_dims(
            tf.range(float(self.rollout - 1), 0, -1), 1) *
         tf.ones([self.rollout - 1, batch_size]) *
         value_estimates[0:1, :],
         value_estimates], 0)
    log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                           log_probs], 0)
    prev_log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                                prev_log_probs], 0)
    relative_log_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                                    relative_log_probs], 0)
    target_values = tf.concat(
        [self.gamma ** tf.expand_dims(
            tf.range(float(self.rollout - 1), 0, -1), 1) *
         tf.ones([self.rollout - 1, batch_size]) *
         target_values[0:1, :],
         target_values], 0)

    sum_rewards = discounted_future_sum(rewards, self.gamma, self.rollout)
    sum_log_probs = discounted_future_sum(log_probs, self.gamma, self.rollout)
    sum_prev_log_probs = discounted_future_sum(prev_log_probs, self.gamma, self.rollout)
    sum_relative_log_probs = discounted_future_sum(
        relative_log_probs, self.gamma, self.rollout)

    if self.use_target_values:
      last_values = shift_values(
          target_values, self.gamma, self.rollout,
          final_target_values)
    else:
      last_values = shift_values(value_estimates, self.gamma, self.rollout,
                                 final_values)

    future_values = (
        - self.tau * sum_log_probs
        - self.eps_lambda * sum_relative_log_probs
        + sum_rewards + last_values)
    baseline_values = value_estimates

    adv = tf.stop_gradient(-baseline_values + future_values)
    if self.clip_adv:
      adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))
    policy_loss = -adv * sum_log_probs
    critic_loss = -adv * (baseline_values - last_values)

    policy_loss = tf.reduce_mean(
        tf.reduce_sum(policy_loss * not_pad, 0))
    critic_loss = tf.reduce_mean(
        tf.reduce_sum(critic_loss * not_pad, 0))

    # loss for gradient calculation
    loss = (self.policy_weight * policy_loss +
            self.critic_weight * critic_loss)

    # actual quantity we're trying to minimize
    raw_loss = tf.reduce_mean(
        tf.reduce_sum(not_pad * adv * (-baseline_values + future_values), 0))

    gradient_ops = self.training_ops(
        loss, learning_rate=self.learning_rate)

    tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0))
    tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0))
    tf.summary.histogram('future_values', future_values)
    tf.summary.histogram('baseline_values', baseline_values)
    tf.summary.histogram('advantages', adv)
    tf.summary.scalar('avg_rewards',
                      tf.reduce_mean(tf.reduce_sum(rewards, 0)))
    tf.summary.scalar('policy_loss',
                      tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
    tf.summary.scalar('critic_loss',
                      tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss))
    tf.summary.scalar('eps_lambda', self.eps_lambda)

    return (loss, raw_loss,
            future_values[self.rollout - 1:, :],
            gradient_ops, tf.summary.merge_all())


class TRPO(ActorCritic):
  """TRPO."""

  def get(self, rewards, pads, values, final_values,
          log_probs, prev_log_probs, target_log_probs,
          entropies, logits,
          target_values, final_target_values, actions=None):
    not_pad = 1 - pads
    batch_size = tf.shape(rewards)[1]

    rewards = not_pad * rewards
    value_estimates = not_pad * values[:, :, 0]
    log_probs = not_pad * sum(log_probs)
    prev_log_probs = not_pad * prev_log_probs
    target_values = not_pad * tf.stop_gradient(target_values)
    final_target_values = tf.stop_gradient(final_target_values)

    sum_rewards = discounted_future_sum(rewards, self.gamma, self.rollout)

    if self.use_target_values:
      last_values = shift_values(
          target_values, self.gamma, self.rollout,
          final_target_values)
    else:
      last_values = shift_values(value_estimates, self.gamma, self.rollout,
                                 final_values)

    future_values = sum_rewards + last_values
    baseline_values = value_estimates


    adv = tf.stop_gradient(-baseline_values + future_values)
    if self.clip_adv:
      adv = tf.minimum(self.clip_adv, tf.maximum(-self.clip_adv, adv))
    policy_loss = -adv * tf.exp(log_probs - prev_log_probs)
    critic_loss = -adv * baseline_values

    policy_loss = tf.reduce_mean(
        tf.reduce_sum(policy_loss * not_pad, 0))
    critic_loss = tf.reduce_mean(
        tf.reduce_sum(critic_loss * not_pad, 0))
    raw_loss = policy_loss

    # loss for gradient calculation
    if self.policy_weight == 0:
      policy_loss = 0.0
    elif self.critic_weight == 0:
      critic_loss = 0.0

    loss = (self.policy_weight * policy_loss +
            self.critic_weight * critic_loss)

    gradient_ops = self.training_ops(
        loss, learning_rate=self.learning_rate)

    tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0))
    tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0))
    tf.summary.scalar('avg_rewards',
                      tf.reduce_mean(tf.reduce_sum(rewards, 0)))
    tf.summary.scalar('policy_loss',
                      tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
    tf.summary.scalar('critic_loss',
                      tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('raw_loss', raw_loss)

    return (loss, raw_loss, future_values,
            gradient_ops, tf.summary.merge_all())


class SparsePCL(PCL):

    def get(self, rewards, pads, values, final_values,
            log_probs, prev_log_probs, target_log_probs,
            entropies, logits,
            target_values, final_target_values, actions=None):
        assert len(logits) == 1, 'only one discrete action allowed'
        assert actions is not None

        not_pad = 1 - pads
        time_length = tf.shape(rewards)[0]
        batch_size = tf.shape(rewards)[1]
        num_actions = tf.shape(logits[0])[2]

        rewards = not_pad * rewards
        value_estimates = not_pad * values[:, :, 0]
        lambda_coefs = tf.exp(1.0 * not_pad * values[:, :, 1])
        Lambda_sigmoid = not_pad * tf.sigmoid(values[:, :, 2])

        #remove final computation

        logits = logits[0][:-1]
        # logits = [logit[:-1] for logit in logits] #logits[:-1]  # [:-1]

        tau_logits = tf.reshape(
            spmax_tau(tf.reshape(logits, [time_length * batch_size, -1])),
            [time_length, batch_size, 1])

        pi_probs = not_pad * tf.reduce_sum(
            tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            -1)

        lambdas = not_pad * tf.reduce_sum(
            tf.nn.relu(tau_logits - logits) * tf.one_hot(actions, num_actions),
            -1)

        Lambdas = Lambda_sigmoid * (-self.tau / 2)

        # Prepend.
        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                             not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
             tf.ones([self.rollout - 1, batch_size]) *
             value_estimates[0:1, :],
             value_estimates], 0)
        lambda_coefs = tf.concat(
            [tf.ones([self.rollout - 1, batch_size]),
             lambda_coefs], 0)
        pi_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs], 0)
        lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             lambdas], 0)
        Lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             Lambdas], 0)

        sum_rewards = discounted_future_sum(rewards + self.tau / 2, self.gamma, self.rollout)
        sum_pi_probs = discounted_future_sum(pi_probs, self.gamma, self.rollout)
        sum_lambdas = discounted_future_sum(lambdas * lambda_coefs, self.gamma, self.rollout)
        sum_Lambdas = discounted_future_sum(Lambdas, self.gamma, self.rollout)

        # last_values = tf.stop_gradient(shift_values(value_estimates, self.gamma, self.rollout))
        last_values = shift_values(value_estimates, self.gamma, self.rollout)

        future_values = (
                - self.tau * sum_pi_probs
                # + self.tau * sum_lambdas
                + sum_lambdas
                - sum_Lambdas
                + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        raw_adv = adv

        policy_loss = -adv * (sum_pi_probs - sum_lambdas + sum_Lambdas)
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            tf.reduce_sum(policy_loss * not_pad, 0))
        critic_loss = tf.reduce_mean(
            tf.reduce_sum(critic_loss * not_pad, 0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            tf.reduce_sum(not_pad * adv * (-baseline_values + future_values), 0))

        gradient_ops = self.training_ops(
            loss, learning_rate=self.learning_rate)

        tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0))
        tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0))
        tf.summary.histogram('future_values', future_values)
        tf.summary.histogram('baseline_values', baseline_values)
        tf.summary.histogram('advantages', adv)
        tf.summary.scalar('avg_rewards',
                          tf.reduce_mean(tf.reduce_sum(rewards, 0)))
        tf.summary.scalar('policy_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('critic_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss))
        tf.summary.scalar('eps_lambda', self.eps_lambda)

        return (loss, raw_loss, future_values,
                gradient_ops, tf.summary.merge_all())

class GeneralSparsePCL(PCL):

    def get(self, rewards, pads, values, final_values,
            log_probs, prev_log_probs, target_log_probs,
            entropies, logits,
            target_values, final_target_values, actions=None):
        assert len(logits) == 1, 'only one discrete action allowed'
        assert actions is not None

        not_pad = 1 - pads
        time_length = tf.shape(rewards)[0]
        batch_size = tf.shape(rewards)[1]
        num_actions = tf.shape(logits[0])[2]

        rewards = not_pad * rewards
        value_estimates = not_pad * values[:, :, 0]
        lambda_coefs = tf.exp(1.0 * not_pad * values[:, :, 1])
        Lambda_sigmoid = not_pad * tf.sigmoid(values[:, :, 2])

        #remove final computation

        logits = logits[0][:-1]#/(self.k * self.q)
        # logits = logits[0][:-1]

        # logits = [logit[:-1] for logit in logits] #logits[:-1]  # [:-1]

        tau_logits = tf.reshape(
            spmax_tau(tf.reshape(logits, [time_length * batch_size, -1])),
            [time_length, batch_size, 1])

        pi_probs = not_pad * tf.reduce_sum(
            tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            -1)

        pi_probs = tf.pow(pi_probs, (self.q - 1))

        lambdas = not_pad * tf.reduce_sum(
            tf.nn.relu(tau_logits - logits) * tf.one_hot(actions, num_actions),
            -1)

        Lambdas = Lambda_sigmoid * (-self.tau * self.k/(self.q - 1))

        # Prepend.
        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                             not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
             tf.ones([self.rollout - 1, batch_size]) *
             value_estimates[0:1, :],
             value_estimates], 0)
        lambda_coefs = tf.concat(
            [tf.ones([self.rollout - 1, batch_size]),
             lambda_coefs], 0)
        pi_probs = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs], 0)
        lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             lambdas], 0)
        Lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             Lambdas], 0)

        sum_rewards = discounted_future_sum(rewards + self.k * self.tau / (self.q - 1), self.gamma, self.rollout)
        sum_pi_probs = discounted_future_sum(pi_probs, self.gamma, self.rollout)
        sum_lambdas = discounted_future_sum(lambdas * lambda_coefs, self.gamma, self.rollout)
        sum_Lambdas = discounted_future_sum(Lambdas, self.gamma, self.rollout)

        # last_values = tf.stop_gradient(shift_values(value_estimates, self.gamma, self.rollout))
        last_values = shift_values(value_estimates, self.gamma, self.rollout)

        future_values = (
                - ((self.tau * self.q * self.k) / (self.q - 1)) * sum_pi_probs
                # + self.tau * sum_lambdas
                + sum_lambdas
                - sum_Lambdas
                + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        raw_adv = adv

        policy_loss = -adv * (sum_pi_probs - sum_lambdas + sum_Lambdas)
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            tf.reduce_sum(policy_loss * not_pad, 0))
        critic_loss = tf.reduce_mean(
            tf.reduce_sum(critic_loss * not_pad, 0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            tf.reduce_sum(not_pad * adv * (-baseline_values + future_values), 0))

        gradient_ops = self.training_ops(
            loss, learning_rate=self.learning_rate)

        tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0))
        tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0))
        tf.summary.histogram('future_values', future_values)
        tf.summary.histogram('baseline_values', baseline_values)
        tf.summary.histogram('advantages', adv)
        tf.summary.scalar('avg_rewards',
                          tf.reduce_mean(tf.reduce_sum(rewards, 0)))
        tf.summary.scalar('policy_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('critic_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss))
        tf.summary.scalar('eps_lambda', self.eps_lambda)

        return (loss, raw_loss, future_values,
                gradient_ops, tf.summary.merge_all())

class GeneralSparsePCLV2(PCL):

    def get(self, rewards, pads, values, final_values,
            log_probs, prev_log_probs, target_log_probs,
            entropies, logits,
            target_values, final_target_values, actions=None):
        assert len(logits) == 1, 'only one discrete action allowed'
        assert actions is not None

        not_pad = 1 - pads
        time_length = tf.shape(rewards)[0]
        batch_size = tf.shape(rewards)[1]
        num_actions = tf.shape(logits[0])[2]

        rewards = not_pad * rewards
        value_estimates = not_pad * values[:, :, 0]
        lambda_coefs = tf.exp(1.0 * not_pad * values[:, :, 1])
        Lambda_sigmoid = not_pad * tf.sigmoid(values[:, :, 2])

        #remove final computation

        # big_o = 0.00001 * (self.q - 2)
        big_o = 1

        logits = logits[0][:-1]#/(self.k * self.q)
        # logits = [logit[:-1] for logit in logits] #logits[:-1]  # [:-1]

        tau_logits = tf.reshape(
            spmax_tau(tf.reshape(logits, [time_length * batch_size, -1])),
            [time_length, batch_size, 1])

        pi_probs_1 = not_pad * tf.reduce_sum(
            # tf.nn.relu(logits - tau_logits + big_o/((self.q-1)*(self.q-1))) * tf.one_hot(actions, num_actions),
            tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            -1)

        pi_probs_2 = not_pad * tf.reduce_sum(
            # tf.nn.relu(logits - tau_logits + big_o/((self.q-1)*(self.q-1))) * tf.one_hot(actions, num_actions),
            tf.nn.relu(logits - tau_logits) * tf.one_hot(actions, num_actions),
            -1)

        lambdas = not_pad * tf.reduce_sum(
            # tf.nn.relu(tau_logits - logits - big_o/((self.q-1)*(self.q-1))) * tf.one_hot(actions, num_actions),
            tf.nn.relu(tau_logits - logits) * tf.one_hot(actions, num_actions),
            -1)

        Lambdas = Lambda_sigmoid * (-2 * self.tau * self.k/(self.q + 1))

        # Prepend.
        not_pad = tf.concat([tf.ones([self.rollout - 1, batch_size]),
                             not_pad], 0)
        rewards = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             rewards], 0)
        value_estimates = tf.concat(
            [self.gamma ** tf.expand_dims(
                tf.range(float(self.rollout - 1), 0, -1), 1) *
             tf.ones([self.rollout - 1, batch_size]) *
             value_estimates[0:1, :],
             value_estimates], 0)
        lambda_coefs = tf.concat(
            [tf.ones([self.rollout - 1, batch_size]),
             lambda_coefs], 0)
        pi_probs_1 = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs_1], 0)

        pi_probs_1 = tf.pow(pi_probs_1, (self.q - 1))

        pi_probs_2 = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                              pi_probs_2], 0)

        pi_probs_2 = tf.pow(pi_probs_2, (self.q + 1))


        lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             lambdas], 0)
        Lambdas = tf.concat([tf.zeros([self.rollout - 1, batch_size]),
                             Lambdas], 0)

        sum_rewards = discounted_future_sum(rewards + self.k * self.tau / (self.q - 1), self.gamma, self.rollout)
        sum_pi_probs_1 = discounted_future_sum(pi_probs_1, self.gamma, self.rollout)
        sum_pi_probs_2 = discounted_future_sum(pi_probs_2, self.gamma, self.rollout)
        sum_lambdas = discounted_future_sum(lambdas * lambda_coefs, self.gamma, self.rollout)
        sum_Lambdas = discounted_future_sum(Lambdas, self.gamma, self.rollout)

        # last_values = tf.stop_gradient(shift_values(value_estimates, self.gamma, self.rollout))
        last_values = shift_values(value_estimates, self.gamma, self.rollout)

        future_values = (
                - ((self.tau * (self.k * self.q)) / (self.q - 1)) * sum_pi_probs_1
                + ((self.tau * self.k * (self.q-1)) / (self.q + 1)) * sum_pi_probs_2
                # + (self.tau * self.k) * sum_lambdas
                + sum_lambdas
                - sum_Lambdas
                + sum_rewards + last_values)
        baseline_values = value_estimates

        adv = tf.stop_gradient(-baseline_values + future_values)
        raw_adv = adv

        policy_loss = -adv * (sum_pi_probs_1 - sum_pi_probs_2 - sum_lambdas + sum_Lambdas)
        critic_loss = -adv * (baseline_values - last_values)

        policy_loss = tf.reduce_mean(
            tf.reduce_sum(policy_loss * not_pad, 0))
        critic_loss = tf.reduce_mean(
            tf.reduce_sum(critic_loss * not_pad, 0))

        # loss for gradient calculation
        loss = (self.policy_weight * policy_loss +
                self.critic_weight * critic_loss)

        # actual quantity we're trying to minimize
        raw_loss = tf.reduce_mean(
            tf.reduce_sum(not_pad * adv * (-baseline_values + future_values), 0))

        gradient_ops = self.training_ops(
            loss, learning_rate=self.learning_rate)

        tf.summary.histogram('log_probs', tf.reduce_sum(log_probs, 0))
        tf.summary.histogram('rewards', tf.reduce_sum(rewards, 0))
        tf.summary.histogram('future_values', future_values)
        tf.summary.histogram('baseline_values', baseline_values)
        tf.summary.histogram('advantages', adv)
        tf.summary.scalar('avg_rewards',
                          tf.reduce_mean(tf.reduce_sum(rewards, 0)))
        tf.summary.scalar('policy_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('critic_loss',
                          tf.reduce_mean(tf.reduce_sum(not_pad * policy_loss)))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('raw_loss', tf.reduce_mean(raw_loss))
        tf.summary.scalar('eps_lambda', self.eps_lambda)

        return (loss, raw_loss, future_values,
                gradient_ops, tf.summary.merge_all())
