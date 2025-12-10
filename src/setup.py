from setuptools import setup

setup(
    name='SToG',
    version='0.1.0',
    description='SToG - Stochastic Gating library for robust feature selection in deep learning',
    long_description_content_type="text/markdown",
    url='https://github.com/intsystems/SToG',
    author='Eynullayev Altay, Firsov Sergey, Rubtsov Denis, Karpeev Gleb',
    author_email='',
    license='MIT',
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'scikit-learn>=0.24.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.12.0',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='feature selection stochastic gating deep learning PyTorch',
    project_urls={
        'Documentation': 'https://intsystems.github.io/SToG/',
        'Source': 'https://github.com/intsystems/SToG'
    },
)