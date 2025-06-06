from setuptools import setup, find_packages

setup(
    name='degrademaster',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'rdkit',
        'tensorboard',
        # add others here
    ],
    entry_points={
        'console_scripts': [
            'degrademaster=degrademaster.main:entry_point',
        ],
    },
    python_requires='>=3.7',
)
