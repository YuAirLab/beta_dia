from setuptools import setup, find_packages

import beta_dia

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='beta_dia',
    version=beta_dia.__version__,
    license='Apache',
    description='A cool project that does something awesome on diaPASEF data.',
    author='Song Jian',
    author_email='songjian2022@suda.edu.cn',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YuAirLab/beta_dia",

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        'h5py',
        'matplotlib',
        'networkx',
        'numba',
        'numpy<2.0.0',
        'pandas',
        'pyzstd',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'pyarrow',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'beta_dia=beta_dia.dist.main:main',
        ],
    },
)