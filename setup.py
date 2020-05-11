import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()


def get_version() -> str:
    version_filepath = os.path.join(os.path.dirname(__file__), 'vivid', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]
    assert False


def get_install_requires():
    install_requires = [
        'tqdm',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy',
        'pandas',
        'joblib',
        'optuna',
        'xgboost',
        'lightgbm',
        'keras',
        'tensorflow',
        'feather-format'
    ]
    return install_requires


def get_extra_requires():
    extras = {
        'test': ['pytest', 'pytest-cov', 'parameterized', 'ipython', 'jupyter', 'notebook', 'tornado==5.1.1', ],
        'document': ['sphinx', 'sphinx_rtd_theme']
    }
    return extras


setup(
    name='vivid',
    version=get_version(),
    author='nyk510',
    packages=find_packages(),
    include_package_data=True,
    license='BSD License',
    description='Support Tools for Machine Learning VIVIDLY',
    long_description=README,
    url='https://atma.co.jp/',
    author_email='yamaguchi@atma.co.jp',
    install_requires=get_install_requires(),
    tests_require=get_extra_requires()['test'],
    extras_require=get_extra_requires(),
    classifiers=[
        'Environment :: CLI Environment',
        'Framework :: scikit-learn',
        'Framework :: scikit-learn :: X.Y',  # replace "X.Y" as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
