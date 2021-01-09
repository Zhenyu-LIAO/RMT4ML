# RMT4ML
This repository contains [`MATLAB`](https://www.mathworks.com/products/matlab.html) and [`Python`](https://www.python.org/) codes for visualizing random matrix theory results and their applications to machine learning, in [Random Matrix Theory for Machine Learning](https://zhenyu-liao.github.io/pdf/RMT4ML.pdf).

In each subfolder (named after the corresponding section) there are:

* a `.html` file containing the [`MATLAB`](https://www.mathworks.com/products/matlab.html) or [IPython Notebook](https://ipython.org/notebook.html) demos
* a `.m` or `.ipynb` source file

* Chapter 1 Introduction
* Chapter 2 Basics of Random Matrix Theory
	* Section 2.1 Fundamental objects
	* Section 2.2 Foundational random matrix results
		* Section 2.2.1 Key lemma and identities: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.2.1/html/lemma_plots.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.2.1/lemma_plots.ipynb)
		* Section 2.2.2 The Marcenko-Pastur and semi-circle laws: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.2.2/html/MP_and_SC.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.2.2/MP_and_SC.ipynb)
		* Section 2.2.3 Large sample covariance matrices and generalized semi-circles: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.2.3/html/SCM_and_DSC.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.2.3/SCM_and_DSC.ipynb)
	* Section 2.3 Advanced spectrum considerations for sample covariances: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.3/html/advanced_spectrum.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.3/advanced_spectrum.ipynb)
		* Section 2.3.1 Limiting spectrum
		* Section 2.3.2 "No eigenvalue outside the support"
	* Section 2.4 Preliminaries on statistical inference
		* Section 2.4.1 Linear eigenvalue statistics: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.4.1/html/linear_eig_stats.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.4.1/linear_eig_stats.ipynb)
		* Section 2.4.2 Eigenvector projections and subspace methods: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.4.2/html/eigenvec_proj.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.4.2/eigenvec_proj.ipynb)
	* Section 2.5 Spiked model: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.5/html/spiked_models.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.5/spiked_models.ipynb)
		* Section 2.5.1 Isolated eigenvalues
		* Section 2.5.2 Isolated eigenvectors
		* Section 2.5.3 Limiting fluctuations
		* Section 2.5.4 Further discussions and other spiked models
	* Section 2.6 Information-plus-noise, deformed Wigner, and other models
	* Section 2.7 Beyond vectors of independent entries: concentration of measure in RMT
	* Section 2.8 Concluding remarks
	* Section 2.9 Exercises
* Chapter 3 Statistical Inference in Linear Models
	* Section 3.1 Detection and estimation in information-plus-noise models
		* Section 3.1.1 GLRT asymptotics [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.1/html/GLRT.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.1/GLRT.ipynb)
		* Section 3.1.2 Linear and Quadratic Discriminant Analysis [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.1/html/LDA.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.1/LDA.ipynb)
		* Section 3.1.1 Subspace methods: the G-MUSIC algorithm [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.1/html/GMUSIC.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.1/GMUSIC.ipynb)
	* Section 3.2 Covariance matrix distance estimation [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.2/html/cov_distance_estimation.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.2/cov_distance_estimation.ipynb)
	* Section 3.3 M-estimator of scatter [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.3/html/M_estim_of_scatter.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.3/M_estim_of_scatter.ipynb)
	* Section 3.4 Concluding remarks
	* Section 3.5 Practical course material: 
		* The Wasserstein distance estimation [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.5/html/Wasserstein_dist.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.5/Wasserstein_dist.ipynb)
		* Robust portfolio optimization via Tyler estimator [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.5/html/robust_portfolio.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.5/robust_portfolio.ipynb)
* Chapter 4 Kernel Methods
	* Section 4.1 Basic setting
	* Section 4.2 Distance and inner-product random kernel matrices [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.2/html/dist_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.2/dist_kernel.ipynb)
	* Section 4.3 The alpha-beta random kernel model [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.3/html/alpha_beta_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.3/alpha_beta_kernel.ipynb)
	* Section 4.4 Properly scaling kernels [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.4/html/proper_scale_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.4/proper_scale_kernel.ipynb)
	* Section 4.5 Implications to kernel methods
		* Section 4.5.1 Application to kernel spectral clustering [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.5/html/kernel_spectral_clustering.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.5/kernel_spectral_clustering.ipynb)
		* Section 4.5.2 Application to semi-supervised kernel learning [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.5/html/semi_supervised_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.5/semi_supervised_kernel.ipynb)
		* Section 4.5.3 Application to kernel ridge regression [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.5/html/kernel_ridge.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.5/kernel_ridge.ipynb)
	* Section 4.6 Concluding remarks
	* Section 4.7 Practical course material: 
		* Complexity-performance trade-off in spectral clustering with sparse kernel [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.7/html/sparse_clustering.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.7/sparse_clustering.ipynb)
		* Towards transfer learning with kernel regression [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.7/html/transfer.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.7/transfer.ipynb)
* Chapter 5 Large Neural Networks
	* Section 5.1 Random neural networks
		* Section 5.1.1 [Regression with random neural networks](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.1/html/random_NN.html)
		* Section 5.1.2 [Delving deeper into limiting kernel](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.1/html/random_feature_GMM.html)
	* Section 5.2 [Gradient descent dynamics in learning linear neural nets](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.2/html/grad_descent_dynamics.html)
	* Section 5.3 [Recurrent neural nets: echo-state works](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.3/html/ESN.html)
	* Section 5.4 Concluding remarks
	* Section 5.5 Practical course material
		* [Effective kernel of large dimensional random Fourier features](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.5/html/random_Fourier.html)
* Chapter 6 Optimization-based Methods with Non-explicit Solutions
	* Section 6.1 [Generalized linear classifier](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/6.1/html/empirical_risk_min.html)
	* Section 6.2 [Large dimensional support vector machines](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/6.2/html/SVM.html)
	* Section 6.3 Concluding remarks
	* Section 6.4 Practical course material
		* [Phase retrieval](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/6.4/html/phase_retrieval.html)
* Chapter 7 Community Detection on Graphs
	* Section 7.1 Community detection in dense graphs
		* Section 7.1.1 [The stochastic block model](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/7.1/html/SBM.html)
		* Section 7.1.2 [The degree-correlated stochastic block model](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/7.1/html/DCSBM.html)
	* Section 7.2 [From dense to sparse graphs: a different approach](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/7.2/html/sparse_graph.html)
	* Section 7.3 Concluding remarks
	* Section 7.4 Practical course material
		* [Gaussian fluctuations of the SBM eigenvectors](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/7.1/html/SBM.html)
* Chapter 8 [Discussions on Universality and Practical Applications](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/8/html/RMT_universality.html)


