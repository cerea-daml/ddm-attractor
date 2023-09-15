from setuptools import find_packages, setup

setup(
    name='ddim_for_attractors',
    packages=find_packages(
        include=["dyn_ddim"]
    ),
    version='0.1.0',
    description='Package for denoising diffusion models of dynamical systems.',
    author='Tobias Finn',
    license='MIT',
)
