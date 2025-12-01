from setuptools import setup, find_packages

setup(
  name = 'diffusion_policy',
  version='0.1.0', 
  packages = find_packages(),
  python_requires='>=3.9',
  install_requires=[              
        'gym==0.21.0', 
    ],
)
