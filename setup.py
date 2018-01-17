"""Package meta-data"""

from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

EXTENSIONS = [Extension('treeagent._build', ['treeagent/_build.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('treeagent._models', ['treeagent/_models.pyx'],
                        include_dirs=[np.get_include()])]

setup(
    name='treeagent',
    version='0.0.1',
    description='Reinforcement Learning with decision trees',
    url='https://github.com/unixpickle/treeagent-py',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='BSD',
    keywords='ai reinforcement learning',
    packages=find_packages(exclude=['examples', 'experiments']),
    ext_modules=cythonize(EXTENSIONS),
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
        'anyrl>=0.10.0,<0.12.0'
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
    }
)
