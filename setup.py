from setuptools import setup

setup(
    name='dp-data',
    version='1.0',
    description='Preprocesses tabular datasets for DP evaluation',
    url='https://github.com/terranceliu/dp-data',
    author='Terrance Liu',
    license='MIT',
    packages=['dp_data'],
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy', 'scikit_learn',
                     'tqdm', 'folktables', 'xgboost', 'lightgbm',],
)