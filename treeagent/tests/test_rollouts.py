"""
Benchmarks for gathering rollouts.
"""

import random
import unittest

from anyrl.envs import batched_gym_env
from anyrl.rollouts import TruncatedRoller
from anyrl.spaces import gym_space_distribution
import gym
import numpy as np

from treeagent.agent import ActorCritic
from treeagent.models import TreeLeaf, TreeBranch, Ensemble

def test_batched_env_rollouts(benchmark):
    """
    Benchmark rollouts with a batched environment and a
    regular truncated roller.
    """
    env = batched_gym_env([lambda: gym.make('Pong-v0')]*8)
    try:
        agent = ActorCritic(gym_space_distribution(env.action_space),
                            gym_space_distribution(env.observation_space))
        agent.actor = _testing_ensemble()
        agent.critic = _testing_ensemble(num_outs=1)
        roller = TruncatedRoller(env, agent, 128)
        with agent.frozen():
            benchmark(roller.rollouts)
    finally:
        env.close()

def _testing_ensemble(num_outs=6, num_trees=7500):
    """
    Generate an ensemble of random trees.
    """
    weights = []
    trees = []
    for _ in range(num_trees):
        trees.append(_random_tree(num_outs))
        weights.append(np.random.normal())
    return Ensemble(np.random.normal(size=(num_outs,)), weights=weights, models=trees)

def _random_tree(num_outs, max_depth=4):
    """
    Create a random tree.
    """
    if max_depth == 0 or random.random() < 0.5:
        return TreeLeaf(np.random.normal(size=(num_outs,)))
    cutoff = random.randrange(0x100)
    return TreeBranch(random.randrange(num_outs), cutoff,
                      _random_tree(num_outs, max_depth=max_depth-1),
                      _random_tree(num_outs, max_depth=max_depth-1))

if __name__ == '__main__':
    unittest.main()
