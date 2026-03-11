from distutils.core import setup
from setuptools import find_packages


setup(
    name='cval_sdk',
    version='0.0.0.1',
    description='python computer vision active learning library',
    author='DGQ | Cyrill Belyakov | Eduard Shamidanov',
    author_email='',
    url='https://cval.ai',
    package_dir={
        '': '.',
    },
    packages=find_packages(include=['cval_sdk', 'cval_sdk.*']),
    install_requires=[
        'pydantic==1.10.9',
        'requests>=2.31.0',
        'loguru==0.7.0',
    ]
)
