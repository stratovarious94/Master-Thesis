from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Thesis2019',
    version='0.0.1',
    description='Product Classification & Product Matching using CNNs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/jonniedask/capstone-project-2019',
    author='Ioannis Daskalopoulos',
    author_email='stratovarious94@gmail.com',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Data Science :: Convolutional Neural Networks',
        'License :: ',
        'Operating System :: Ubuntu LTS 18.04'
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='data science deep learning product classification matching',
    packages=find_packages(),
    python_requires='>=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    install_requires=[
        "tensorflow-gpu==2.0.0-beta1",
        "pandas==0.25.0",
        "sklearn==0.0",
        "pillow==6.1.0",
        "matplotlib==3.1.1",
        "seaborn==0.9.0"
    ]
)
