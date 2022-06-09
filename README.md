# RMT4ML
This repository contains [`MATLAB`](https://www.mathworks.com/products/matlab.html) and [`Python`](https://www.python.org/) codes for visualizing random matrix theory results and their applications to machine learning, in [Random Matrix Theory for Machine Learning](https://zhenyu-liao.github.io/pdf/RMT4ML.pdf).

In each subfolder (named after the corresponding section) there are:

* a `.html` file containing the [`MATLAB`](https://www.mathworks.com/products/matlab.html) or [IPython Notebook](https://ipython.org/notebook.html) demos
* a `.m` or `.ipynb` source file

* Chapter 1 Introduction
	* Section 1.1 Motivation: The Pitfalls of Large-Dimensional Statistics
	* Section 1.2 Random Matrix Theory as an Answer
	* Section 1.3 Outline and Online Toolbox
* Chapter 2 Random Matrix Theory
	* Section 2.1 Fundamental Objects
	* Section 2.2 Foundational Random Matrix Results
		* Section 2.2.1 Key Lemma and Identities: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.2/html/lemma_plots.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.2/lemma_plots.ipynb)
		* Section 2.2.2 The Marcenko-Pastur and Semicircle Laws: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.2/html/MP_and_SC.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.2/MP_and_SC.ipynb)
		* Section 2.2.3 Large-Dimensional Sample Covariance Matrices and Generalized Semicircles: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.2/html/SCM_and_DSC.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.2/SCM_and_DSC.ipynb)
	* Section 2.3 Advanced Spectrum Considerations for Sample Covariances: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.3/html/advanced_spectrum.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.3/advanced_spectrum.ipynb)
	* Section 2.4 Preliminaries on Statistical Inference
		* Section 2.4.1 Linear Eigenvalue Statistics: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.4/html/linear_eig_stats.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.4/linear_eig_stats.ipynb)
		* Section 2.4.2 Eigenvector Projections and Subspace Methods: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.4/html/eigenvec_proj.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.4/eigenvec_proj.ipynb)
	* Section 2.5 Spiked Models: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/2.5/html/spiked_models.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/2.5/spiked_models.ipynb)
	* Section 2.6 Information-plus-Noise, Deformed Wigner, and Other Models
	* Section 2.7 Beyond Vectors of Independent Entries: Concentration of Measure in RMT
	* Section 2.8 Concluding Remarks
	* Section 2.9 Exercises
* Chapter 3 Statistical Inference in Linear Models
	* Section 3.1 Detection and Estimation in Information-plus-Noise Models
		* Section 3.1.1 GLRT Asymptotics: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.1/html/GLRT.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.1/GLRT.ipynb)
		* Section 3.1.2 Linear and Quadratic Discriminant Analysis: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.1/html/LDA.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.1/LDA.ipynb)
		* Section 3.1.1 Subspace Methods: The G-MUSIC Algorithm: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.1/html/GMUSIC.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.1/GMUSIC.ipynb)
	* Section 3.2 Covariance Matrix Distance Estimation: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.2/html/cov_distance_estimation.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.2/cov_distance_estimation.ipynb)
	* Section 3.3 M-Estimators of Scatter: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.3/html/M_estim_of_scatter.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.3/M_estim_of_scatter.ipynb)
	* Section 3.4 Concluding Remarks
	* Section 3.5 Practical Course Material: 
		* The Wasserstein distance estimation: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.5/html/Wasserstein_dist.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.5/Wasserstein_dist.ipynb)
		* Robust portfolio optimization via Tyler estimator: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/3.5/html/robust_portfolio.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/3.5/robust_portfolio.ipynb)
* Chapter 4 Kernel Methods
	* Section 4.1 Basic Setting
	* Section 4.2 Distance and Inner-Product Random Kernel Matrices
		* Section 4.2.1 Main Intuitions 
		* Section 4.2.2 Main Results: Distance Random Kernel Matrices: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.2/html/dist_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.2/dist_kernel.ipynb)
		* Section 4.2.3 Motivation: $\alpha-\beta$ Random Kernel Matrices 
		* Section 4.2.4 Main Results: $\alpha-\beta$ Random Kernel Matrices: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.2/html/alpha_beta_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.2/alpha_beta_kernel.ipynb)
	* Section 4.3 Properly Scaling Kernel Model: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.3/html/proper_scale_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.3/proper_scale_kernel.ipynb)
	* Section 4.4 Implications to Kernel Methods
		* Section 4.4.1 Application to Kernel Spectral Clustering: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.4/html/kernel_spectral_clustering.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.4/kernel_spectral_clustering.ipynb)
		* Section 4.4.2 Application to Semi-supervised Kernel Learning: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.4/html/semi_supervised_kernel.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.4/semi_supervised_kernel.ipynb)
		* Section 4.4.3 Application to Kernel Ridge Regression: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.4/html/kernel_ridge.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.4/kernel_ridge.ipynb)
		* Section 4.4.4 Summary of Section 4.4
	* Section 4.5 Concluding Remarks
	* Section 4.6 Practical Course Material
		* Complexity-performance trade-off in spectral clustering with sparse kernel: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.6/html/sparse_clustering.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.6/sparse_clustering.ipynb)
		* Towards transfer learning with kernel regression: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/4.6/html/transfer.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/4.6/transfer.ipynb)
* Chapter 5 Large Neural Networks
	* Section 5.1 Random Neural Networks
		* Section 5.1.1 Regression with Random Neural Networks: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.1/html/random_NN.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/5.1/random_NN.ipynb)
		* Section 5.1.2 Delving Deeper into Limiting Kernels: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.1/html/random_feature_GMM.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/5.1/random_feature_GMM.ipynb)
	* Section 5.2 Gradient Descent Dynamics in Learning Linear Neural Nets: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.2/html/grad_descent_dynamics.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/5.2/grad_descent_dynamics.ipynb)
	* Section 5.3 Recurrent Neural Nets: Echo-State Networks: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.3/html/ESN.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/5.3/ENS.ipynb)
	* Section 5.4 Concluding Remarks
	* Section 5.5 Practical Course Material: performance of large-dimensional random Fourier features [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/5.5/html/random_Fourier.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/5.5/html/random_Fourier.ipynb)
* Chapter 6 Large-Dimensional Convex Optimization
	* Section 6.1 Generalized Linear Classifier: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/6.1/html/empirical_risk_min.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/6.1/empirical_risk_min.ipynb)
	* Section 6.2 Large-Dimensional Support Vector Machines
	* Section 6.3 Concluding Remarks
	* Section 6.4 Practical Course Material: phase retrieval [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/6.4/html/phase_retrieval.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/6.4/phase_retrieval.ipynb)
* Chapter 7 Community Detection on Graphs
	* Section 7.1 Community Detection in Dense Graphs
		* Section 7.1.1 The Stochastic Block Model: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/7.1/html/SBM.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/7.1/SBM.ipynb)
		* Section 7.1.2 The Degree-Correlated Stochastic Block Model: 
		[Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/7.1/html/DCSBM.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/7.1/DCSBM.ipynb)
	* Section 7.2 From Dense to Sparse Graphs: A Different Approach:
	[Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/7.2/html/sparse_graph.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/7.2/sparse_graph.ipynb)
	* Section 7.3 Concluding Remarks
	* Section 7.4 Practical Course Material: Asymptotic Gaussian fluctuations of the SBM dominant eigenvector
* Chapter 8 Universality and Real Data: [Matlab code](https://htmlpreview.github.io/?https://github.com/Zhenyu-LIAO/RMT4ML/blob/master/8/html/RMT_universality.html) and [Python code](https://nbviewer.jupyter.org/github/Zhenyu-LIAO/RMT4ML/blob/master/8/RMT_universality.ipynb)
	* Section 8.1 From Gaussian Mixtures to Concentrated Random Vectors and GAN Data
	* Section 8.2 Wide-Sense Universality in Large-Dimensional Machine Learning
	* Section 8.3 Discussions and Conclusions