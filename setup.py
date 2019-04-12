#!/usr/bin/env/python

from setuptools import setup, find_packages

setup(
    name='distil-auto-ml',
    author='Ben Johnson',
    author_email='bkj.322@gmail.com',
    classifiers=[],
    description='distil automl server',
    keywords=['automl'],
    license='ALV2',
    packages=find_packages(),
    version="0.1.0",
    dependency_links=[
        'git+https://github.com/uncharted-distil/distil-primitives.git#egg=DistilPrimitives',
    ]
)
