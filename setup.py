from setuptools import setup

setup(
    name='DRInC',
    version='0.1.0',
    description='Distributionally Robust Infinite-horizon Control',
    url='https://github.com/DecodEPFL/DRInC/tree/jsb_dev',
    author='Jean-Sebastien Brouillon',
    author_email='jean-sebastien.brouillon@epfl.ch',
    license='CC BY 4.0',
    packages=['DRInC'],
    install_requires=['cvxpy',
                      'numpy',
                      'mosek',
                      'scipy',
                      'matplotlib',
                      'tqdm'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
