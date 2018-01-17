"""
Tests for tree building.

This is mostly intended as a check to make sure that tree
optimizations do not break or change building outcomes.
"""

import unittest

import gym
import numpy as np

from treeagent.build import build_tree
from treeagent.models import TreeLeaf, TreeBranch

class TestBuildSimpleCases(unittest.TestCase):
    """
    Simple test cases for tree building.
    """
    def test_predictable_split(self):
        """
        Test that a predictable split actually occurs.
        """
        inputs, outputs = _predictable_data(20000)
        node = build_tree(inputs, outputs, min_leaf=5, max_depth=1)
        expected_leaves = [[-10, 7], [-7, -10]]
        self.assertIsInstance(node, TreeBranch)
        self.assertIsInstance(node.less_than, TreeLeaf)
        self.assertIsInstance(node.greater_equal, TreeLeaf)
        self.assertEqual(node.split_feature, 1)
        self.assertTrue(np.allclose(node.threshold, 5, rtol=1e-2, atol=1e-2))
        for child, expected in zip([node.less_than, node.greater_equal], expected_leaves):
            self.assertTrue(np.allclose(child.output, expected,
                                        rtol=1e-2, atol=1e-2))

class TestFloatBuildRegression(unittest.TestCase):
    """
    Make sure building trees with a bunch of canned float
    data produces the same results in the future.
    """
    def test_leaf(self):
        """
        Test building with no branches.
        """
        inputs, outputs = regression_data()
        tree = build_tree(inputs, outputs, min_leaf=3, max_depth=0)
        self.assertIsInstance(tree, TreeLeaf)
        self.assertTrue(np.allclose(tree.output, np.mean(outputs, axis=0)))

    def test_one_branch(self):
        """
        Test building with one branch.
        """
        inputs, outputs = regression_data()
        tree = build_tree(inputs, outputs, min_leaf=3, max_depth=1)
        self.assertIsInstance(tree, TreeBranch)
        self.assertIsInstance(tree.less_than, TreeLeaf)
        self.assertIsInstance(tree.greater_equal, TreeLeaf)
        self.assertEqual(tree.split_feature, 6)
        self.assertTrue(np.allclose(tree.threshold, 0.696461959))
        self.assertTrue(np.allclose(tree.less_than.output,
                                    [0.15382855, -0.82658136, -0.04233113]))
        self.assertTrue(np.allclose(tree.greater_equal.output,
                                    [0.0946953, 0.30802816, 0.39473894]))

    def test_two_branches(self):
        """
        Test building with two branches.
        """
        inputs, outputs = regression_data()
        tree = build_tree(inputs, outputs, min_leaf=6, max_depth=2)
        self.assertIsInstance(tree, TreeBranch)
        self.assertIsInstance(tree.less_than, TreeBranch)
        self.assertIsInstance(tree.greater_equal, TreeLeaf)
        self.assertIsInstance(tree.less_than.less_than, TreeLeaf)
        self.assertIsInstance(tree.less_than.greater_equal, TreeLeaf)
        self.assertEqual(tree.split_feature, 6)
        self.assertTrue(np.allclose(tree.threshold, 0.696461959))
        self.assertTrue(np.allclose(tree.greater_equal.output,
                                    [0.0946953, 0.30802816, 0.39473894]))
        self.assertEqual(tree.less_than.split_feature, 6)
        self.assertTrue(np.allclose(tree.less_than.threshold, -0.6259399115000001))
        self.assertTrue(np.allclose(tree.less_than.less_than.output,
                                    [0.66664267, -0.3186112, -0.53174299]))
        self.assertTrue(np.allclose(tree.less_than.greater_equal.output,
                                    [-0.14531307, -1.12289727, 0.24315912]))

    def test_two_branches_permuted(self):
        """
        Test that the features from test_two_branches
        remain valid when the data is permuted.
        """
        inputs, outputs = regression_data()
        inputs[:, 3] = inputs[:, 6]
        inputs[:, 6] = inputs[:, 1]
        tree = build_tree(inputs, outputs, min_leaf=6, max_depth=2)
        self.assertEqual(tree.split_feature, 3)
        self.assertEqual(tree.less_than.split_feature, 3)

    def test_threading_equiv(self):
        """
        Make sure that threading does not influence the
        structure of the trees.
        """
        inputs, outputs = regression_data()
        tree1 = build_tree(inputs, outputs, min_leaf=6, max_depth=3, num_threads=1)
        tree2 = build_tree(inputs, outputs, min_leaf=6, max_depth=3, num_threads=4)
        tree3 = build_tree(inputs, outputs, min_leaf=6, max_depth=3, num_threads=12)
        self.assertTrue(_trees_equivalent(tree1, tree2))
        self.assertTrue(_trees_equivalent(tree2, tree3))

class TestUint8BuildRegression(unittest.TestCase):
    """
    Make sure building trees with a bunch of canned uint8
    data produces the same results in the future.
    """
    def test_leaf(self):
        """
        Test building with no branches.
        """
        inputs, outputs = uint8_regression_data()
        tree = build_tree(inputs, outputs, min_leaf=3, max_depth=0)
        self.assertIsInstance(tree, TreeLeaf)
        self.assertTrue(np.allclose(tree.output, np.mean(outputs, axis=0)))

    def test_one_branch(self):
        """
        Test building with one branch.
        """
        inputs, outputs = uint8_regression_data()
        tree = build_tree(inputs, outputs, min_leaf=3, max_depth=1)
        self.assertIsInstance(tree, TreeBranch)
        self.assertIsInstance(tree.less_than, TreeLeaf)
        self.assertIsInstance(tree.greater_equal, TreeLeaf)
        self.assertEqual(tree.split_feature, 5)
        self.assertTrue(np.allclose(tree.threshold, 126.5))
        self.assertTrue(np.allclose(tree.less_than.output,
                                    [0.70325124, -0.03628322, 0.33606252]))
        self.assertTrue(np.allclose(tree.greater_equal.output,
                                    [-0.43895862, -0.78483254, -0.10020673]))

    def test_two_branches(self):
        """
        Test building with two branches.
        """
        inputs, outputs = uint8_regression_data()
        tree = build_tree(inputs, outputs, min_leaf=6, max_depth=2)
        self.assertIsInstance(tree, TreeBranch)
        self.assertIsInstance(tree.less_than, TreeBranch)
        self.assertIsInstance(tree.greater_equal, TreeBranch)
        self.assertIsInstance(tree.greater_equal.less_than, TreeLeaf)
        self.assertIsInstance(tree.greater_equal.greater_equal, TreeLeaf)
        self.assertIsInstance(tree.less_than.less_than, TreeLeaf)
        self.assertIsInstance(tree.less_than.greater_equal, TreeLeaf)
        self.assertEqual(tree.split_feature, 5)
        self.assertTrue(np.allclose(tree.threshold, 126.5))
        self.assertEqual(tree.less_than.split_feature, 0)
        self.assertTrue(np.allclose(tree.less_than.threshold, 153.5))
        self.assertEqual(tree.greater_equal.split_feature, 6)
        self.assertTrue(np.allclose(tree.greater_equal.threshold, 157))
        self.assertTrue(np.allclose(tree.less_than.less_than.output,
                                    [0.05919572, 0.0693915, 0.65720719]))
        self.assertTrue(np.allclose(tree.less_than.greater_equal.output,
                                    [1.13262165, -0.10673304, 0.12196609]))
        self.assertTrue(np.allclose(tree.greater_equal.less_than.output,
                                    [0.05945089, -1.83148921, -0.04628649]))
        self.assertTrue(np.allclose(tree.greater_equal.greater_equal.output,
                                    [-0.77123165, -0.0870612, -0.13615353]))

    def test_threading_equiv(self):
        """
        Make sure that threading does not influence the
        structure of the trees.
        """
        inputs, outputs = uint8_regression_data()
        tree1 = build_tree(inputs, outputs, min_leaf=6, max_depth=3, num_threads=1)
        tree2 = build_tree(inputs, outputs, min_leaf=6, max_depth=3, num_threads=4)
        tree3 = build_tree(inputs, outputs, min_leaf=6, max_depth=3, num_threads=12)
        self.assertTrue(_trees_equivalent(tree1, tree2))
        self.assertTrue(_trees_equivalent(tree2, tree3))

def _trees_equivalent(tree1, tree2):
    """
    Check that two trees are similar.
    """
    if isinstance(tree1, TreeLeaf) != isinstance(tree2, TreeLeaf):
        return False
    if isinstance(tree1, TreeLeaf):
        return np.allclose(tree1.output, tree2.output)
    return (np.allclose(tree1.threshold, tree2.threshold) and
            tree1.split_feature == tree2.split_feature and
            _trees_equivalent(tree1.less_than, tree2.less_than) and
            _trees_equivalent(tree1.greater_equal, tree2.greater_equal))

def test_atari_pong(benchmark):
    """
    Benchmark training on Pong-v0 rollouts.
    """
    _atari_benchmark(benchmark, feature_frac=0.01, num_threads=1)

def test_atari_pong_full_threaded(benchmark):
    """
    Benchmark training on Pong-v0 rollouts with all the
    features and all the available CPU cores.
    """
    _atari_benchmark(benchmark)

def _atari_benchmark(benchmark, **build_kwargs):
    """
    Run a Pong-v0 benchmark.
    """
    env = gym.make('Pong-v0')
    inputs = []
    outputs = []
    env.reset()
    for _ in range(300):
        obs, _, done, _ = env.step(0)
        if done:
            env.reset()
        inputs.append(np.array(obs).flatten())
        outputs.append(np.random.normal(size=6))
    env.close()
    benchmark(build_tree, inputs, outputs, **build_kwargs)

def test_mujoco_reacher(benchmark):
    """
    Benchmark training on Reacher-v1 rollouts.
    """
    env = gym.make('Reacher-v1')
    inputs = []
    outputs = []
    env.reset()
    for _ in range(300):
        obs, _, done, _ = env.step(np.random.normal(size=2))
        if done:
            env.reset()
        inputs.append(np.array(obs).flatten())
        outputs.append(np.random.normal(size=2))
    env.close()
    benchmark(build_tree, inputs, outputs, num_threads=1)

def _predictable_data(num_samples):
    """
    Produce a dataset that induces a split on feature
    1 at mean value 5, a mean left leaf of [-10, 7],
    and a mean right leaf of [-7, -10].
    """
    left_obs = np.random.rand(num_samples, 3) + np.array([0, 3.95, 0])
    right_obs = np.random.rand(num_samples, 3) + np.array([0, 5.05, 0])
    left_out = np.random.rand(num_samples, 2) + np.array([-10.5, 6.5])
    right_out = np.random.rand(num_samples, 2) + np.array([-7.5, -10.5])
    joined_obs = left_obs.tolist() + right_obs.tolist()
    joined_out = left_out.tolist() + right_out.tolist()
    return joined_obs, joined_out

def regression_data():
    """
    Get a sizable dataset of inputs and outputs for
    regression tests.
    """
    # pylint: disable=C0301
    data = np.array([[-2.28686293e-02, -9.21355008e-01, -1.26078237e+00, 1.75762545e-01, -8.35929145e-02, 5.16535648e-01, 1.11812576e+00],
                     [-1.88750995e+00, 9.23060606e-02, -3.15614884e-01, -2.08864474e+00, -4.67986383e-01, -9.91662435e-01, -1.67728486e+00],
                     [-4.45935784e-01, -7.27133199e-01, -5.43062436e-01, -4.13440110e-01, -2.09560016e+00, 7.15704960e-02, 9.72928035e-01],
                     [5.85636685e-01, -8.55386457e-01, 9.27219460e-01, 5.68193735e-01, 4.24598917e-01, -2.28163891e+00, -1.24275523e+00],
                     [4.99041330e-01, -1.90843143e+00, 5.56198951e-01, 5.00400564e-01, 5.63316186e-01, 1.15575832e+00, 2.48742413e-01],
                     [1.36075705e+00, -4.65438271e-01, 1.11719272e+00, -1.03055891e+00, 3.98048934e-01, 6.87061155e-01, 1.16521902e+00],
                     [-5.73466263e-01, -2.02685202e+00, 1.26827251e+00, -7.68573073e-01, 5.42322813e-01, 7.13415444e-01, 2.01506021e+00],
                     [-1.34469640e+00, 8.34168993e-01, -1.83007808e+00, 9.76324998e-02, 1.89758190e-01, 4.25766612e-01, 7.48760271e-01],
                     [-8.15009657e-01, 3.45170927e-01, -5.17842487e-01, -3.37594828e+00, -1.62429610e-01, -2.63284556e+00, -3.34618344e-01],
                     [-4.54895343e-01, 8.04672521e-01, -1.02358987e+00, -3.67218540e-01, -5.19560782e-01, -1.07149313e+00, -1.05378257e+00],
                     [3.93877628e-01, -1.22704009e+00, -1.33443711e+00, -7.62261882e-01, 7.78468922e-01, -1.49031531e-01, 5.80642607e-01],
                     [-8.72961355e-01, 1.42560403e+00, 9.59853800e-01, 3.27768420e-01, 1.47748015e+00, 1.11212588e-02, -8.41695388e-01],
                     [1.44665023e+00, -8.82692540e-02, -3.55517003e-01, -2.03287628e+00, 1.10205819e+00, 2.68081634e-01, -7.76386129e-01],
                     [2.10178972e-01, -1.57700988e+00, 1.27788580e-01, -1.99974480e+00, 6.66864494e-01, -9.13745646e-01, 1.02973011e+00],
                     [1.63845731e+00, -2.04787532e+00, -1.64150734e+00, -8.90378605e-02, 7.73840765e-01, -1.80987894e+00, 1.48903036e-01],
                     [1.71384718e-01, -8.48528682e-01, 8.29979159e-01, -6.95294553e-01, -1.79726385e-01, 1.64937798e+00, 1.27985770e+00],
                     [-1.50429130e+00, 3.64157041e-01, 1.69742125e+00, -1.11850821e+00, -1.18719211e+00, 5.33873852e-01, 8.48516220e-02],
                     [-2.19081732e+00, 3.42202808e-01, -4.68382966e-01, 6.13891741e-01, -7.89466769e-01, 1.43136820e+00, -9.12879762e-01],
                     [9.28322247e-01, 2.10146690e+00, 1.33861859e+00, -7.90064710e-02, 6.97800287e-02, 1.13953778e+00, 8.75319075e-01],
                     [-2.46040879e+00, -8.90575631e-01, 8.73687001e-01, 1.06355836e-01, -5.20105552e-01, -4.31921336e-01, -1.18900851e+00],
                     [-1.70026285e+00, -1.48502213e-02, -2.61277642e-01, 1.56986706e+00, 6.49095705e-01, 2.06011906e-01, -1.14120039e-01],
                     [-7.92033818e-01, -1.37786967e+00, 1.01796725e+00, 2.10795694e-01, 8.75457934e-01, 5.92108714e-02, -3.20849660e-02],
                     [-1.67160362e+00, -2.76183821e+00, 2.81288584e-01, 1.49960365e+00, 4.35872550e-01, 1.69960747e+00, 1.79397571e-01],
                     [1.64071899e+00, -6.35172074e-01, -2.20273168e+00, -1.09274016e+00, 8.10068063e-04, -1.27693442e-01, -4.75493694e-01],
                     [-1.21797583e+00, -3.35516698e-01, 1.64903948e+00, -1.11693733e-02, -6.94477889e-01, 9.22994404e-01, 1.47298485e+00],
                     [1.68215112e+00, -4.90059720e-01, 1.53154721e-01, -1.29763072e+00, -5.48730362e-01, 7.28734283e-01, 8.48370363e-01],
                     [1.18720798e-01, -1.42551088e+00, -2.07973355e-01, -1.19640910e+00, -1.63486017e-01, 4.70039495e-01, 1.21666443e+00],
                     [-1.78563754e-01, 1.49679432e+00, 1.76255553e-01, 8.78543500e-02, -2.14541252e-01, 4.55542049e-01, 6.44163647e-01],
                     [1.22558882e+00, -1.76468795e+00, 1.37380980e+00, 2.47202072e-01, 1.27350106e+00, 1.51274274e-02, -1.09699346e-01],
                     [-1.09316269e+00, -3.68502284e-01, 1.57943930e+00, 9.69993308e-02, 9.43457928e-01, -4.00548258e-01, -4.47379478e-01]])
    outputs = data[list(range(15, 30)) + list(range(15))][:, 2:5]
    return data, outputs

def uint8_regression_data():
    """
    Get regression test data with uint8 observations.
    """
    obs = np.array([[183, 105, 113, 25, 12, 114, 100],
                    [216, 132, 144, 253, 65, 36, 124],
                    [144, 121, 201, 246, 107, 70, 59],
                    [155, 161, 69, 93, 78, 217, 237],
                    [224, 177, 51, 15, 187, 118, 15],
                    [0, 48, 35, 80, 165, 154, 252],
                    [211, 186, 94, 180, 171, 27, 246],
                    [53, 144, 213, 67, 38, 37, 53],
                    [206, 130, 208, 14, 58, 135, 196],
                    [163, 108, 151, 90, 140, 114, 89],
                    [203, 223, 244, 16, 31, 135, 12],
                    [150, 229, 254, 82, 63, 221, 120],
                    [152, 214, 169, 170, 89, 144, 180],
                    [224, 6, 164, 244, 190, 65, 225],
                    [123, 233, 60, 196, 119, 101, 187],
                    [191, 127, 84, 80, 81, 162, 254],
                    [170, 109, 106, 168, 99, 242, 22],
                    [134, 154, 125, 39, 251, 202, 185],
                    [203, 71, 160, 137, 27, 57, 69],
                    [166, 61, 179, 206, 150, 35, 98],
                    [149, 73, 250, 33, 108, 246, 111],
                    [197, 200, 223, 212, 239, 19, 30],
                    [178, 96, 210, 196, 234, 165, 221],
                    [69, 22, 103, 211, 121, 226, 22],
                    [115, 152, 135, 255, 42, 246, 234],
                    [200, 88, 153, 168, 216, 182, 253],
                    [46, 170, 156, 41, 59, 95, 231],
                    [62, 134, 139, 81, 235, 27, 1],
                    [144, 48, 251, 217, 228, 167, 134],
                    [82, 52, 131, 117, 15, 37, 199]], dtype='uint8')
    return obs, regression_data()[1]

if __name__ == '__main__':
    unittest.main()
