# RMT4ML
This repository contains [MATLAB](https://www.mathworks.com/products/matlab.html) for visualizing random matrix theory results and their applications to large dimensional machine learning problems, in the preprint of [Random Matrix Advances in Large Dimensional Machine Learning](https://zhenyu-liao.github.io/pdf/RMT4ML.pdf).

In each subfolder (named after the corresponding section number) there are:

* a HTML file containing the MATLAB demos in the section
* some .m functions containing the (improved) algorithms in the section

Below is the table of content of the book that links to corresponding [MATLAB](https://www.mathworks.com/products/matlab.html) simulations.

* Chapter 1 Introduction
* Chapter 2 Basics of Random Matrix Theory
	* Section 2.1 Fundamental objects
	* Section 2.2 Foundational random matrix results
		* Section 2.2.1 [Key lemma and identities](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/2.2.1/html/lemma_plots.html)
		* Section 2.2.2 [The Marcenko-Pastur and semi-circle laws](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/2.2.2/html/MP_and_SC.html)
		* Section 2.2.3 [Large sample covariance matrices and generalized semi-circles](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/2.2.3/html/SCM_and_DSC.html)
	* Section 2.3 [Advanced spectrum considerations for sample covariances](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/2.3/html/advanced_spectrum.html)
		* Section 2.3.1 Limiting spectrum
		* Section 2.3.2 "No eigenvalue outside the support"
	* Section 2.4 Preliminaries on statistical inference
		* Section 2.4.1 [Linear eigenvalue statistics](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/2.4.1/html/linear_eig_stats.html)
		* Section 2.4.2 [Eigenvector projections and subspace methods](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/2.4.2/html/eigenvec_proj.html)
	* Section 2.5 [Spiked model](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/2.5/html/spiked_models.html)
		* Section 2.5.1 Isolated eigenvalues
		* Section 2.5.2 Isolated eigenvectors
		* Section 2.5.3 Limiting fluctuations
		* Section 2.5.4 Further discussions and other spiked models
	* Section 2.6 Information-plus-noise, deformed Wigner, and other models
	* Section 2.7 Beyond vectors of independent entries: concentration of measure in RMT
	* Section 2.8 Concluding remarks
	* Section 2.9 [Exercises]()
* Chapter 3 Statistical Inference in Linear Models
	* Section 3.1 Detection and estimation in information-plus-noise models
		* Section 3.1.1 [GLRT asymptotics](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/3.1/html/GLRT.html)
		* Section 3.1.2 [Linear and Quadratic Discriminant Analysis](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/3.1/html/LDA.html)
		* Section 3.1.1 [Subspace methods: the G-MUSIC algorithm](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/3.1/html/GMUSIC.html)
	* Section 3.2 [Covariance matrix distance estimation](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/3.2/html/cov_distance_estimation.html)
	* Section 3.3 [M-estimator of scatter](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/3.3/html/M_estim_of_scatter.html)
	* Section 3.4 Concluding remarks
	* Section 3.5 Practical course material: 
		* [The Wasserstein distance estimation](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/3.5/html/Wasserstein_dist.html)
		* [Robust portfolio optimization via Tyler estimator](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/3.5/html/robust_portfolio.html)
* Chapter 4 Kernel Methods
	* Section 4.1 Basic setting
	* Section 4.2 [Distance and inner-product random kernel matrices](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.2/html/dist_kernel.html)
	* Section 4.3 [The alpha-beta random kernel model](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.3/html/alpha_beta_kernel.html)
	* Section 4.4 [Properly scaling kernels](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.4/html/proper_scale_kernel.html)
	* Section 4.5 Implications to kernel methods
		* Section 4.5.1 [Application to kernel spectral clustering](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.5/html/kernel_spectral_clustering.html)
		* Section 4.5.2 [Application to semi-supervised kernel learning](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.5/html/semi_supervised_kernel.html)
		* Section 4.5.3 [Application to kernel ridge regression](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.5/html/kernel_ridge.html)
	* Section 4.6 Concluding remarks
	* Section 4.7 Practical course material: 
		* [Complexity-performance trade-off in spectral clustering with sparse kernel](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.7/html/sparse_clustering.html)
		* [Towards transfer learning with kernel regression](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/4.7/html/transfer.html)
* Chapter 5 Large Neural Networks
	* Section 5.1 Random neural networks
		* Section 5.1.1 [Regression with random neural networks](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/5.1/html/random_NN.html)
		* Section 5.1.2 [Delving deeper into limiting kernel](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/5.1/html/random_feature_GMM.html)
	* Section 5.2 [Gradient descent dynamics in learning linear neural nets](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/5.2/html/grad_descent_dynamics.html)
	* Section 5.3 [Recurrent neural nets: echo-state works](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/5.3/html/ESN.html)
	* Section 5.4 Concluding remarks
	* Section 5.5 Practical course material
		* [Effective kernel of large dimensional random Fourier features](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/5.5/html/random_Fourier.html)
* Chapter 6 Optimization-based Methods with Non-explicit Solutions
	* Section 6.1 [Generalized linear classifier](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/6.1/html/empirical_risk_min.html)
	* Section 6.2 [Large dimensional support vector machines](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/6.2/html/SVM.html)
	* Section 6.3 Concluding remarks
	* Section 6.4 Practical course material
		* [Phase retrieval](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/6/html/phase_retrieval.html)
* Chapter 7 Community Detection on Graphs
	* Section 7.1 Community detection in dense graphs
		* Section 7.1.1 [The stochastic block model](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/7.1/html/SBM.html)
		* Section 7.1.2 [The degree-correlated stochastic block model](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/7.1/html/DCSBM.html)
	* Section 7.2 [From dense to sparse graphs: a different approach](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/7.2/html/sparse_graph.html)
	* Section 7.3 Concluding remarks
	* Section 7.4 Practical course material
		* [Gaussian fluctuations of the SBM eigenvectors](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/Matlab_resource/7.4/html/Gaussian_eigenvector.html)
* Chapter 8 [Discussions on Universality and Practical Applications]()


