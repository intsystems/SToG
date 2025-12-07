==========
References
==========

This section lists academic references for the methods implemented in SToG.

Primary References
==================

Stochastic Gating (STG)
~~~~~~~~~~~~~~~~~~~~~~~

.. [Yamada2020] Yamada, Y., Lindenbaum, O., Negahban, S., & Kluger, Y. (2020).
   "Feature Selection using Stochastic Gates."
   In *International Conference on Machine Learning (ICML)* (pp. 10648-10659).
   https://proceedings.mlr.press/v119/yamada20a/yamada20a.pdf

   - Original stochastic gating method
   - Gaussian-based continuous relaxation
   - Foundational work for the library

Straight-Through Estimator (STE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [Bengio2013] Bengio, Y., Léonard, N., & Courville, A. (2013).
   "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation."
   *arXiv preprint* arXiv:1308.0850.
   https://arxiv.org/abs/1308.0850

   - Gradient approximation through discrete operations
   - Enables backpropagation through binarization

.. [Courbariaux2015] Courbariaux, M., Bengio, Y., & David, J. P. (2015).
   "Binarized Neural Networks."
   In *Advances in Neural Information Processing Systems (NIPS)* (pp. 3123-3131).
   https://arxiv.org/abs/1602.02830

   - Application of STE to binarized networks
   - Practical implementation details

Gumbel-Softmax
~~~~~~~~~~~~~~~

.. [Jang2017] Jang, E., Gu, S., & Poole, B. (2017).
   "Categorical Reparameterization with Gumbel-Softmax."
   In *International Conference on Learning Representations (ICLR)*.
   https://arxiv.org/abs/1611.01144

   - Gumbel trick for categorical distribution
   - Temperature-annealed softmax

.. [Maddison2017] Maddison, C. J., Hoffman, M. D., & Mnih, A. (2017).
   "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables."
   In *International Conference on Learning Representations (ICLR)*.
   https://arxiv.org/abs/1611.00712

   - Concrete (Gumbel-Softmax) distribution
   - Continuous relaxation theory

Correlated Features
~~~~~~~~~~~~~~~~~~~

.. [FeatureSelection2020] Yuan, M., Lin, Y., & Bi, W. (2010).
   "Robust Estimation of Structured Sparsity."
   In *The Annals of Statistics*, 38(6), 3242-3274.
   https://doi.org/10.1214/10-AOS815

   - Group sparsity for correlated features
   - Structured variable selection

.. [Jiang2017] Jiang, B., Wang, S., & Zhu, S. (2017).
   "Data Poisoning Attacks Against Multi-Party Machine Learning."
   In *35th International Conference on Machine Learning (ICML)*.

   - Handling correlated feature groups

Feature Selection Overview
==========================

.. [Guyon2003] Guyon, I., & Elisseeff, A. (2003).
   "An introduction to variable and feature selection."
   *Journal of machine learning research*, 3(Mar), 1157-1182.
   http://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf

   - Comprehensive feature selection survey
   - Classical methods and evaluation

.. [Kohavi1997] Kohavi, R., & John, G. H. (1997).
   "Wrappers for feature subset selection."
   *Artificial intelligence*, 97(1-2), 273-324.
   https://doi.org/10.1016/S0004-3702(97)00043-X

   - Wrapper methods for feature selection
   - Cross-validation approaches

Related Deep Learning Methods
==============================

.. [Louizos2018] Louizos, C., Welling, M., & Kingma, D. P. (2018).
   "Learning sparse neural networks through L0 regularization."
   In *International Conference on Learning Representations (ICLR)*.
   https://arxiv.org/abs/1712.01312

   - L0 regularization for neural network pruning
   - Related to feature selection

.. [Tibshirani1996] Tibshirani, R. (1996).
   "Regression shrinkage and selection via the lasso."
   *Journal of the royal statistical society: series B*, 58(1), 267-288.
   https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

   - LASSO and L1 regularization
   - Classical sparse regression method

.. [Loh2015] Loh, P. L., & Wainwright, M. J. (2015).
   "Regularized M-estimators with nonconvexity: Statistical and algorithmic theory for compound losses."
   *The Journal of Machine Learning Research*, 16(1), 559-604.

   - Theoretical foundations of sparse estimation
   - High-dimensional statistics

Related Benchmarking Work
==========================

.. [Demšar2006] Demšar, J. (2006).
   "Statistical comparisons of classifiers over multiple data sets."
   *The Journal of Machine Learning Research*, 7, 1-30.
   http://www.jmlr.org/papers/volume7/demsargensemble06a/dems...

   - Statistical comparison methodology
   - Benchmark evaluation protocols

Implementation References
==========================

**PyTorch Documentation**
   https://pytorch.org/docs/stable/

**NumPy Documentation**
   https://numpy.org/doc/stable/

**scikit-learn Documentation**
   https://scikit-learn.org/stable/

Citing SToG
===========

If you use SToG in your research, please cite:

.. code-block:: bibtex

   @software{stog2025,
     title={SToG: Stochastic Gating for Feature Selection},
     author={Eynullayev, A. and Rubtsov, D. and Firsov, S. and Karpeev, G.},
     year={2025},
     url={https://github.com/intsystems/SToG}
   }
