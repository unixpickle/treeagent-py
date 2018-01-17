"""
Training algorithms for building up ensembles.
"""

from abc import ABC, abstractmethod

from anyrl.algos import GAE
from anyrl.algos.ppo import clipped_objective
import numpy as np
import tensorflow as tf

from .build import build_tree
from .models import Ensemble

# pylint: disable=R0903
class ACTrainer(ABC):
    """
    Base class for actor-critic trainers.
    """
    def __init__(self,
                 session,
                 actor_critic,
                 adv_est=GAE(lam=0.95, discount=0.99, target_lam=1)):
        self.session = session
        self.actor_critic = actor_critic
        self._adv_est = adv_est

        action_dist = actor_critic.action_dist
        param_shape = (None,) + action_dist.param_shape
        self._action_params_ph = tf.placeholder(tf.float32, shape=param_shape)
        self._old_action_params_ph = tf.placeholder(tf.float32, shape=param_shape)
        self._actions_ph = tf.placeholder(tf.float32,
                                          shape=(None,)+action_dist.out_shape)
        self._advantages_ph = tf.placeholder(tf.float32, shape=(None,))

    # pylint: disable=R0913
    def value_update(self,
                     rollouts,
                     step_size,
                     num_steps,
                     recompute_outs=False,
                     log_fn=None,
                     **build_kwargs):
        """
        Generate an update direction for the value function.

        Parameters:
          rollouts: the batch of rollouts to learn on
          step_size: scale for each tree
          num_steps: number of trees
          log_fn: if set, called with string arguments
            with information about learning
          recompute_outs: if set, don't assume that the
            value estimations in the rollouts came from
            the current value function.
          build_kwargs: arguments passed to build_tree()

        Returns:
          An Ensemble representing the update direction.
        """
        inputs, targets = self._value_batch(rollouts)
        if recompute_outs:
            outputs = self.actor_critic.critic.apply_batch(inputs)
        else:
            outputs = np.array([out['values']
                                for r in rollouts
                                for out in r.step_model_outs])
        ensemble = Ensemble(np.zeros(1, dtype='float32'))
        for i in range(num_steps):
            if log_fn:
                mse = np.mean(np.square(outputs - targets))
                log_fn('batch %d: mse=%f' % (i, mse))
            tree = build_tree(inputs, targets-outputs, **build_kwargs)
            ensemble.models.append(tree)
            ensemble.weights.append(step_size)
            outputs += tree.apply_batch(inputs) * step_size
        return ensemble

    def _policy_batch(self, rollouts):
        """
        Create a batch of policy training samples.

        Returns:
          A tuple (inputs, outputs, actions, advs)
        """
        shaped_advs = self._adv_est.advantages(rollouts)
        outputs, actions, advs = [], [], []
        for rollout, sub_advs in zip(rollouts, shaped_advs):
            outputs.extend([x['action_params'][0] for x in rollout.step_model_outs])
            actions.extend([x['actions'][0] for x in rollout.step_model_outs])
            advs.extend(sub_advs)
        return (_flat_obs_vecs(rollouts),
                np.array(outputs), actions, np.array(advs))

    def _value_batch(self, rollouts):
        """
        Create a batch of value training samples.

        Returns:
          A tuple (inputs, targets)
        """
        targets = [x for r in self._adv_est.targets(rollouts) for x in r]
        return (_flat_obs_vecs(rollouts),
                np.array(targets, dtype='float32').reshape((-1, 1)))

    def _feed_dict(self, orig_outputs, actions, advs):
        """
        Generate a TF feed_dict for the objective.

        Does not include self._action_params_ph.
        """
        return {
            self._old_action_params_ph: orig_outputs,
            self._actions_ph: self.actor_critic.action_dist.to_vecs(actions),
            self._advantages_ph: advs
        }

    def _log_probs(self):
        """
        Get (old_log_probs, new_log_probs) as Tensors.
        """
        action_dist = self.actor_critic.action_dist
        old_log_probs = action_dist.log_prob(self._old_action_params_ph,
                                             self._actions_ph)
        new_log_probs = action_dist.log_prob(self._action_params_ph,
                                             self._actions_ph)
        return (old_log_probs, new_log_probs)

class A2C(ACTrainer):
    """
    A gradient boosted advantage actor-critic trainer.

    Builds step ensembles that approximate the functional
    gradient of the A2C objective.
    """
    # pylint: disable=R0913
    def __init__(self,
                 session,
                 actor_critic,
                 init_step_size=0.1,
                 target_kl=0.01,
                 max_diff_ratio=1.5,
                 adjust_rate=1.25,
                 **ac_kwargs):
        super(A2C, self).__init__(session, actor_critic, **ac_kwargs)
        self.target_kl = target_kl
        self.max_diff_ratio = max_diff_ratio
        self.adjust_rate = adjust_rate
        self._current_step_size = init_step_size

        old_logs, new_logs = self._log_probs()
        objective = tf.reduce_sum(tf.exp(new_logs - old_logs) * self._advantages_ph)
        self._grad = tf.gradients(objective, self._action_params_ph)[0]

        dist = self.actor_critic.action_dist
        kl_div = dist.kl_divergence(self._old_action_params_ph, self._action_params_ph)
        self._mean_kl = tf.reduce_mean(kl_div)

        self._mean_entropy = tf.reduce_mean(dist.entropy(self._action_params_ph))

    # pylint: disable=R0914
    def policy_update(self,
                      rollouts,
                      step_size,
                      num_steps,
                      log_fn=None,
                      **build_kwargs):
        """
        Use gradient boosting to approximate the policy
        gradient.

        Parameters:
          rollouts: the batch of rollouts to learn on
          step_size: scale for each tree
          num_steps: number of trees
          log_fn: if set, called with string arguments
            with information about learning
          build_kwargs: arguments passed to build_tree()

        Returns:
          An Ensemble representing the update direction.
        """
        inputs, orig_outputs, actions, advs = self._policy_batch(rollouts)
        feed_dict = self._feed_dict(orig_outputs, actions, advs)
        feed_dict[self._action_params_ph] = orig_outputs
        grad = np.array(self.session.run(self._grad, feed_dict=feed_dict))

        num_params = np.prod(self.actor_critic.action_dist.param_shape)
        ensemble = Ensemble(np.zeros(num_params, dtype='float32'))

        # Can happen if we get no reward.
        if np.linalg.norm(grad) == 0:
            return ensemble

        cur_output = np.zeros(grad.shape, dtype='float32')
        for i in range(num_steps):
            if log_fn:
                mse = np.mean(np.square(cur_output - grad))
                log_fn('batch %d: mse=%f' % (i, mse))
            tree = build_tree(inputs,
                              (grad - cur_output).reshape((len(grad), -1)),
                              **build_kwargs)
            ensemble.models.append(tree)
            ensemble.weights.append(step_size)
            cur_output += tree.apply_batch(inputs).reshape(grad.shape) * step_size

        self._adapt_step(orig_outputs, cur_output)

        if log_fn:
            log_fn('step result: kl=%f step=%f entropy=%f' %
                   (self._compute_mean_kl(orig_outputs, cur_output),
                    self._current_step_size,
                    self._compute_entropy(orig_outputs, cur_output)))

        ensemble.scale(self._current_step_size)
        return ensemble

    def _adapt_step(self, orig_outputs, step):
        """
        Adjust the step size to try to meet the KL target
        for the step.
        """
        compute_kl = lambda: self._compute_mean_kl(orig_outputs, step)
        while compute_kl() < self.target_kl*self.max_diff_ratio:
            self._current_step_size *= self.adjust_rate
        while compute_kl() > self.target_kl*self.max_diff_ratio:
            self._current_step_size /= self.adjust_rate

    def _compute_mean_kl(self, orig_outputs, step):
        """
        Compute the mean KL divergence by taking the
        ensemble step with the current step size.
        """
        feed_dict = {
            self._old_action_params_ph: orig_outputs,
            self._action_params_ph: orig_outputs + step*self._current_step_size
        }
        return self.session.run(self._mean_kl, feed_dict=feed_dict)

    def _compute_entropy(self, orig_outputs, step):
        """
        Compute the entropy after the step.
        """
        feed_dict = {
            self._action_params_ph: orig_outputs + step*self._current_step_size
        }
        return self.session.run(self._mean_entropy, feed_dict=feed_dict)

# pylint: disable=R0902,R0903
class PPO(ACTrainer):
    """
    An abstract proximal policy optimization technique.

    Subclasses of PPO implement the objective itself.
    """
    # pylint: disable=R0913
    def __init__(self,
                 session,
                 actor_critic,
                 entropy_reg=0.01,
                 **ac_kwargs):
        super(PPO, self).__init__(session, actor_critic, **ac_kwargs)

        action_dist = self.actor_critic.action_dist
        entropy = action_dist.entropy(self._action_params_ph)
        policy_term = self._policy_objective()
        actor_objective = policy_term + entropy_reg * tf.reduce_sum(entropy)
        self._action_grad = tf.gradients(actor_objective, self._action_params_ph)[0]
        self._mean_entropy = tf.reduce_mean(entropy)
        self._mean_objective = policy_term / tf.cast(tf.shape(entropy)[0], tf.float32)
        self._mask = self._sample_mask()

    # pylint: disable=R0914
    def policy_update(self,
                      rollouts,
                      step_size,
                      num_steps,
                      off_policy=False,
                      log_fn=None,
                      **build_kwargs):
        """
        Use policy gradient boosting to compute an update
        direction.

        Parameters:
          rollouts: the batch of rollouts to learn on
          step_size: scale for each tree
          num_steps: number of trees
          log_fn: if set, called with string arguments
            with information about learning
          off_policy: if set, don't assume that the action
            parameters in the rollouts came from the
            current policy.
          build_kwargs: arguments passed to build_tree()

        Returns:
          An Ensemble representing the update direction.
        """
        inputs, orig_outputs, actions, advs = self._policy_batch(rollouts)
        if off_policy:
            raw_outputs = self.actor_critic.actor.apply_batch(inputs)
            outputs = raw_outputs.reshape(orig_outputs.shape)
        else:
            outputs = orig_outputs.copy()
        feed_dict = self._feed_dict(orig_outputs, actions, advs)
        num_params = np.prod(self.actor_critic.action_dist.param_shape)
        ensemble = Ensemble(np.zeros(num_params, dtype='float32'))
        for i in range(num_steps):
            feed_dict[self._action_params_ph] = outputs
            terms = (self._action_grad, self._mask, self._mean_entropy, self._mean_objective)
            grad, mask, entropy, objective = self.session.run(terms, feed_dict=feed_dict)

            mask = np.array(mask)
            if np.count_nonzero(mask) == 0:
                break

            targets = np.array(grad, dtype='float32')
            targets = targets.reshape((targets.shape[0], -1))
            tree = build_tree(inputs[mask], targets[mask], **build_kwargs)
            ensemble.models.append(tree)
            ensemble.weights.append(step_size)
            outputs += tree.apply_batch(inputs).reshape(outputs.shape) * step_size
            if log_fn:
                log_fn('batch %d: objective=%f entropy=%f masked=%d' %
                       (i, objective, entropy, len(mask)-np.count_nonzero(mask)))
        return self._finish_policy_update(ensemble, orig_outputs, outputs)

    # pylint: disable=R0201,W0613
    def _finish_policy_update(self, ensemble, orig_outputs, outputs):
        """
        Called at the end of a policy update.

        You may want to override this for adaptive
        hyper-parameters, ensemble scaling, etc.
        """
        return ensemble

    @abstractmethod
    def _policy_objective(self):
        """
        Compute a quantity to maximize, given the actions,
        action probabilities, original probabilities, etc.

        This quantity should be a sum rather than a mean.
        """
        pass

    def _sample_mask(self):
        """
        Creat a boolean Tensor indicating which samples to
        train on.

        Default behavior is to train on all samples.
        """
        return tf.ones(tf.shape(self._advantages_ph), dtype=tf.bool)

def _flat_obs_vecs(rollouts):
    """
    Get a flattened batch of observations that can be fed
    directly into a Model.
    """
    inputs = [obs for r in rollouts for obs in r.step_observations]
    return np.array(inputs).reshape((len(inputs), -1))

class ClippedPPO(PPO):
    """
    PPO with the clipped objective.
    """
    def __init__(self, session, actor_critic, epsilon=0.1, mask_clipped=False,
                 **ppo_kwargs):
        self.epsilon = epsilon
        self.mask_clipped = mask_clipped
        super(ClippedPPO, self).__init__(session, actor_critic, **ppo_kwargs)

    def _policy_objective(self):
        old_logs, new_logs = self._log_probs()
        raw_obj = clipped_objective(new_logs, old_logs,
                                    self._advantages_ph, self.epsilon)
        return tf.reduce_sum(raw_obj)

    def _sample_mask(self):
        if not self.mask_clipped:
            return super(ClippedPPO, self)._sample_mask()
        old_logs, new_logs = self._log_probs()
        ratio = tf.exp(old_logs - new_logs)
        clipped_ratio = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)
        return ratio*self._advantages_ph <= clipped_ratio*self._advantages_ph

class KLPenaltyPPO(PPO):
    """
    PPO with the adaptive KL penalty.
    """
    # pylint: disable=R0913
    def __init__(self,
                 session,
                 actor_critic,
                 init_beta=0.1,
                 target_kl=0.01,
                 max_diff_ratio=1.5,
                 adjust_rate=2,
                 **ppo_kwargs):
        self.beta = init_beta
        self.target_kl = target_kl
        self.max_diff_ratio = max_diff_ratio
        self.adjust_rate = adjust_rate
        self._beta_ph = tf.placeholder(tf.float32)
        self._kl_term = None
        self._current_beta = init_beta
        super(KLPenaltyPPO, self).__init__(session, actor_critic, **ppo_kwargs)

    def _policy_objective(self):
        if not self._kl_term:
            dist = self.actor_critic.action_dist
            self._kl_term = dist.kl_divergence(self._old_action_params_ph,
                                               self._action_params_ph)
        old_logs, new_logs = self._log_probs()
        ratio = tf.exp(new_logs - old_logs)
        return tf.reduce_sum(ratio * self._advantages_ph -
                             self._beta_ph * self._kl_term)

    def _feed_dict(self, orig_outputs, actions, advs):
        result = super(KLPenaltyPPO, self)._feed_dict(orig_outputs, actions, advs)
        result[self._beta_ph] = self._current_beta
        return result

    def _finish_policy_update(self, ensemble, orig_outputs, outputs):
        feed_dict = {
            self._old_action_params_ph: orig_outputs,
            self._action_params_ph: outputs,
        }
        kl_value = np.mean(self.session.run(self._kl_term, feed_dict=feed_dict))
        if kl_value > self.target_kl*self.max_diff_ratio:
            self._current_beta *= self.adjust_rate
        elif kl_value < self.target_kl/self.max_diff_ratio:
            self._current_beta /= self.adjust_rate
        return ensemble
