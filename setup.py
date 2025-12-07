from setuptools import setup, find_packages

setup(
    name='stog',
    version='0.0.1',
    description='Stochastic Gating for Feature Selection',
    author='Eynullayev A., Rubtsov D., Firsov S., Karpeev G.',
    author_email='',
    url='https://github.com/intsystems/SToG',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
