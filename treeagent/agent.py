"""
RL agents, fully compatible with anyrl.
"""

import anyrl.models
import numpy as np

from .models import Ensemble, FastEnsemble

class ActorCritic(anyrl.models.Model):
    """
    An RL model that uses one model to predict values and
    another to select actions.
    """
    def __init__(self, action_dist, obs_vectorizer, actor=None, critic=None):
        """
        Create an actor-critic model.

        Parameters:
          action_dist: an anyrl.spaces.Distribution for
            sampling actions.
          obs_vectorizer: an anyrl.spaces.Vectorizer for
            processing observations.
          actor: the policy model. Defaults to an empty
            ensemble with zero outputs.
          critic: the value function model. Defaults to an
            ensemble with zero outputs.
        """
        self.action_dist = action_dist
        self.obs_vectorizer = obs_vectorizer
        if actor is None:
            num_params = np.prod(self.action_dist.param_shape)
            self.actor = Ensemble(np.zeros((num_params,), dtype='float32'))
        else:
            self.actor = actor
        if critic is None:
            self.critic = Ensemble(np.zeros((1,), dtype='float32'))
        else:
            self.critic = critic

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        obs_shape = (len(observations), -1)
        obs_vecs = self.obs_vectorizer.to_vecs(observations).reshape(obs_shape)
        raw_act_params = self.actor.apply_batch(obs_vecs)
        action_params = raw_act_params.reshape((-1,) + self.action_dist.param_shape)
        pred_vals = self.critic.apply_batch(obs_vecs).flatten()
        return {
            'action_params': action_params,
            'actions': self.action_dist.sample(action_params),
            'states': None,
            'values': pred_vals
        }

    def frozen(self):
        """
        Produce a context manager that temporarily sets
        the actor and critic to FastEnsembles.

        Example usage:

            with actor_critic.frozen():
                rollouts = roller.rollouts()
            # train here.

        """
        return _FastContext(self)

# pylint: disable=R0903
class _FastContext:
    """
    A context manager for temporarily using FastEnsembles in
    in an ActorCritic.
    """
    def __init__(self, actor_critic):
        self._actor_critic = actor_critic
        self._old_actor = None
        self._old_critic = None

    def __enter__(self):
        self._old_actor = self._actor_critic.actor
        self._old_critic = self._actor_critic.critic
        if isinstance(self._old_actor, Ensemble):
            self._actor_critic.actor = FastEnsemble(self._old_actor)
        if isinstance(self._old_critic, Ensemble):
            self._actor_critic.critic = FastEnsemble(self._old_critic)

    def __exit__(self, _, value, traceback):
        self._actor_critic.actor = self._old_actor
        self._actor_critic.critic = self._old_critic
