# Bayesian Optimization Using Gaussian Process Regression

An implementation of Gaussian-Process based Bayesian Optimization, from scratch, demonstrated with two examples:  

- Learning a demo function composed of multiple sine waves. (`basic_GP.py`).  
- Tuning neural network hyperparameters on MNIST (`nn_hyperparameter_tune_GP.py`).

### What is Bayesian Optimization?

Bayesian Optimization is a technique for efficiently maximizing (or minimizing) an objective function that is computationally expensive to evaluate. It models the function using a surrogate (a Gaussian process in this case) and strategically samples points for evaluation by maximizing an acquisition function, such as Expected Improvement. The expected improvement function (EI) estimates the expected value of improvement over the best result observed so far, based on the surrogate's predictions at a given point. By maximizing this function, we select a point for evaluation which we believe will have the highest chance of exceeding the previously found max. After executing this evaluation, our Gaussian Process becomes a stronger surrogate model since future predictions are conditioned on one more data point. Evidently, the use of a surrogate model and an acquisition function enables significant reduction in the required number of evaluations with the expensive, black-box function since it selects evaluation points in a principled and informed manner.

### What is a Gaussian Process?

As mentioned previously, our Bayesian Optimization implementation uses a Gaussian Process as the surrogate model. Specifically, we use Gaussian Process Regression which models the mapping between inputs and outputs using a multivariate Gaussian distribution. This distribution represents a probability over functions that pass through the different data points (input-output pairs) that the Gaussian Process is conditioned on. In the case of the NN hyperparameter tuning demo, the inputs are network hyperparameter vectors and the output is the accuracy obtained by training the network using those hyperparameters. Let X denote the list of hyperparameter vectors, and Y denote the corresponding list of accuracies. The method `get_pred_mean_var()` of the `Gaussian_Process` class, provides the predictive mean and variance at a test point, conditioned on X and Y: $P(y_{\*} \mid x_{\*}, X, Y)$, where $x_{\*}$ is the test hyperparameter vector and $y_{\*}$ is the predicted accuracy. The implementation is based on Algorithm 2.1 from Gaussian Processes for Machine Learning by Rasmussen and Williams.

Gaussian Process Regression assumes that Y is distributed under a Gaussian with mean of 0 and a covariance of $K(X,X) + \sigma_n^2I$, where K is a covariance function, $\sigma_n^2$ is the noise variance and $I$ is the identity matrix. We will be using the radial basis function (RBF) covariance. This includes two hyperparameters of its own: the lengthscale and signal variance. Altogether, there are three covariance related hyperparameters that we need to tune: lengthscale, signal variance and noise variance. We do this by maximizing the probability of obtaining Y given X: $P(Y|X, \theta)$, where $\theta$ = [lengthscale, signal variance, noise variance]. This is referred to as the marginal likelihood. Algorithm 2.1 of the Gaussian Process for Machine Learning book provides the formula for the log marginal likelihood, which we compute in the `get_log_marginal_likelihood()` method of the `Gaussian_Process` class. 

### A Few Implementational Details:

The Gaussian process makes predictions, conditioned on evaluated points. Before it can be used as a surrogate and provide guidance on which points to evaluate, we must evaluate a few randomly chosen, 'space-filling' points so the Gaussian process has some signal to work with. We then maximize the Expected Improvement function (which uses the Gaussian process for predictions) using the L-BFGS-B algorithm and evaluate the original function at this point. We also maximize the log marginal likelihood using L-BFGS-B. For both the expected improvement and log marginal likelihood, we use Pytorch's autograd to compute the derivative, as opposed to having it approximated by Scipy's L-BFGS-B function.

## Results

### Sinusoidal Demo

In `basic_GP.py`, we try to maximize a simple sinusoidal function. The function is inexpensive to compute, so the use of Bayesian Optimization with a Gaussian Process surrogate is not warranted, but it serves as an easily understandable demonstration of the algorithm.
The video below shows how the Gaussian Process's predictions evolve over time as points chosen by maximizing the Expected Improvement function are evaluated, and added to the list of points the GP is conditioned on.



https://github.com/user-attachments/assets/8afe8a31-9eb8-456d-94ac-08bb29e1939d




### Neural Network Hyperparameter Tuning For MNIST

`nn_hyperparameter_tune_GP.py` includes a `build_model()` function that returns a Tensorflow model based on 7 hyperparameters: number of layers, number of hidden units, learning rate, l1 regularization rate, l2 regularization rate, dropout rate and the hidden layer decrease fraction. (The hidden layer decrease fraction specifies the factor by which the number of hidden units in successive layers is reduced, enabling a gradual tapering of the model architecture.) When an evaluation at a certain point in input space (a set of hyperparameters) is carried out, a model is created using `build_model()` and is then trained on 50000 MNIST images (validated on 10000 images) for 200 epochs. If the validation accuracy doesn't increase for 5 epochs, we end training early. The max validation accuracy of the training run is the $y$ value, and what we attempt to maximize using Bayesian Optimization with a Gaussian process surrogate.

Initially, 30 random, space-filling points were evaluated, before Bayesian Optimization was run for 55 steps. We intended on running Bayesian Optimization for 200 steps, but the Google Colab notebook shut down after the GPU usage limit had been reached.


<img width="600" alt="colab_gp_bo" src="https://github.com/user-attachments/assets/d84ca29a-5d41-45a9-a583-85593411daf7" />

<img width="600" alt="colab_gp_bo_clipped" src="https://github.com/user-attachments/assets/666231e4-688b-450f-9115-8f6cd3bb00f6" />

The 30 random, space-filling points are in blue, while the points selected and evaluated during Bayesian Optimization are in orange. In the second graph, accuracies are clipped at 0.9 to improve visualization.

The maximum accuracy achieved during random sampling was 98.06% while the max accuracy achieved through Bayesian Optimization was 98.41%. An increase of 0.35%, although seemingly small, is quite notable when considering the already high baseline performance of MLPs on MNIST. 22 out of 30 randomly chosen networks achieved an accuracy of over 96%, and 17 out of 30 achieved an accuracy over 97%. After reaching a high level of performance, it is increasingly difficult to squeeze out further performance gains, making this 0.35% improvement especially valuable. If we were to have carried out 200 steps, as intended, we might have been able to obtain an even larger improvement.

One will notice that the accuracies over the Bayesian Optimization steps are quite noisy, with lots of oscillation, and this is very much expected.
As can also be seen in the video of the sinusoidal demo, although we are using Bayesian Optimization to maximize a black-box function, the values at evaluated points do not have an upward trend (as is expected for gradient descent). Instead, the values oscillate as the Expected Improvement function attempts to balance exploration and exploitation in the input space.
