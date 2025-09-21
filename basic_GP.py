import torch
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import numpy as np
import imageio

class Gaussian_Process:
    def __init__(self):
        torch.set_default_dtype(torch.float64)

        #dummy function that we'll try to learn
        self.func = lambda x: torch.sin(3*x) + 0.5*torch.sin(7*x) + 0.3*torch.sin(15*x)
        #3 space-filling starter points. Generally chosen randomly, but we handpick them for this dummy problem
        self.Xs = torch.tensor([1.0,3.0,5.0], dtype=torch.get_default_dtype())
        self.Ys = self.func(self.Xs)
        
        #mean center Ys
        self.Y_mean = self.Ys.mean(0, keepdim=True)
        self.Ys = self.Ys - self.Y_mean
        
        #radial basis function covariance
        #for further detials, see: https://www.cs.toronto.edu/~duvenaud/cookbook/
        self.noise_variance = 0.5 * torch.var(self.Ys) #start at 0.5 of variance of y
        self.signal_variance = 0.5 * torch.var(self.Ys) #start at 0.5 of variance of y
        self.length = 1 #we'll start at one
        #we unsqueeze x1 and x2 at the first and zeroth dimension, respectively, since we want 
        # pairwise squared distances, across all possible pairs of x1 and x2
        self.covariance = lambda x1, x2: self.signal_variance * torch.exp(
            -0.5*(x1.unsqueeze(1)-x2.unsqueeze(0))**2/(self.length**2)
        )
        
        self.bounds = [(0.0,2*math.pi)] #we want to learn the function between 0 and 2pi
        #separate lower bounds and upper bounds
        self.lb, self.ub = zip(*self.bounds)
        self.lb = torch.tensor(self.lb, dtype=torch.get_default_dtype())
        self.ub = torch.tensor(self.ub, dtype=torch.get_default_dtype())
        
        #length, signal_variance, noise variance bounds
        self.cov_param_bounds = [(math.log(1e-5), math.log(1e5)), 
                                 (math.log(1e-5), math.log(1e5)), 
                                 (math.log(1e-5), math.log(1e5))
                                ]
        self.cov_param_lb, self.cov_param_ub = zip(*self.cov_param_bounds)
        self.cov_param_lb = torch.tensor(self.cov_param_lb, dtype=torch.get_default_dtype())
        self.cov_param_ub = torch.tensor(self.cov_param_ub, dtype=torch.get_default_dtype())
        
        #Run the l-bfgs-b algorithm with 30 different random starting points
        # since l-bfgs-b is susceptible to local optima
        self.num_restarts = 30
        
        self.num_train_steps = 50
        
    def get_pred_mean_var(self, test_x):
        #implements part of Algorithm 2.1 of the Gaussian Process for Machine learning book
        #Computing: P(y_*|test_x,X,Y)
        
        #hard to invert k so we calculate kx = y, x = k^-1 * y, using cholesky decomposition
        # instead of calculating k^-1
        
        cov_fn = self.covariance
        
        k = cov_fn(self.Xs, self.Xs)
        jitter = 1e-6 * torch.eye(self.Xs.shape[0], dtype=k.dtype)
        L = torch.linalg.cholesky(k + self.noise_variance * torch.eye(self.Xs.shape[0]) + jitter)
      
        #forward substitution
        forward_sub = torch.linalg.solve_triangular(L, self.Ys.unsqueeze(-1), upper=False)
        #backward substitution
        alpha = torch.linalg.solve_triangular(L.T, forward_sub, upper=True)
        
        k_star = cov_fn(self.Xs, test_x)
        
        pred_mean = torch.matmul(k_star.T, alpha) #predictive mean
        
        v = torch.linalg.solve_triangular(L, k_star, upper=False)
        pred_var = cov_fn(test_x, test_x) - torch.matmul(v.T,v) #predictive variance
       
        return pred_mean, pred_var
    
    def get_log_marginal_likelihood(self, cov_params):
        #this implementation follows Algorithm 2.1 of the Gaussian Process for Machine learning book
        #computing P(Y|X, lengthscale, signal_variance, noise_variance) 
        
        #cov_params should be in log space at this point
        
        cov_params = torch.exp(cov_params)
        length = cov_params[0]
        signal_variance = cov_params[1]
        noise_variance = cov_params[2]

        #new covariance function using the provided cov_params as opposed to the global params
        cov_fn = lambda x1, x2: signal_variance * torch.exp(
            -0.5*(x1.unsqueeze(1)-x2.unsqueeze(0))**2/(length**2)
        )
      
        k = cov_fn(self.Xs, self.Xs)
        jitter = 1e-6 * torch.eye(self.Xs.shape[0])
        L = torch.linalg.cholesky(k + noise_variance * torch.eye(self.Xs.shape[0]) + jitter)
        
        #forward substitution
        forward_sub = torch.linalg.solve_triangular(L, self.Ys.unsqueeze(-1), upper=False)
        #backward substitution
        alpha = torch.linalg.solve_triangular(L.T, forward_sub, upper=True)
        
        log_marginal_likelihood = (-0.5*torch.matmul(self.Ys.unsqueeze(-1).T, alpha) 
                                   - torch.log(L.diagonal()).sum() 
                                   - (self.Xs.shape[0]/2)*torch.log(torch.tensor(2*math.pi)))
      
        return log_marginal_likelihood
    
    def expected_improvement(self, test_x):
        #implementation of the expected improvement function
        #see: https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
        
        pred_mean, pred_var = self.get_pred_mean_var(test_x)
        
        expected_diff = pred_mean - torch.max(self.Ys)
        
        sigma = torch.sqrt(torch.clamp(pred_var, min=1e-9))
        z = expected_diff / sigma
        
        normal = torch.distributions.Normal(0,1)
        ei = expected_diff*normal.cdf(z) + sigma*torch.exp(normal.log_prob(z))
      
        return ei
    
    #get derivative of the expected improvement w.r.t. test point
    def ei_grad(self, test_x):
        test_x.requires_grad_()
        improvement = self.expected_improvement(test_x)
        improvement.backward()
        return test_x.grad
    
    #get derivative of the log marginal likelihood w.r.t. covariance kernel parameters
    def marg_likelihood_grad(self, cov_params):
        cov_params.requires_grad_()
        out = self.get_log_marginal_likelihood(cov_params)
        out.backward()
        return cov_params.grad
    
    def rand_start_point(self):
        point = self.lb + (self.ub - self.lb) * torch.rand(len(self.bounds))
        return point
    
    def rand_cov_param(self):
        point = self.cov_param_lb + (self.cov_param_ub - self.cov_param_lb) * torch.rand(len(self.cov_param_bounds))
        return point
    
    #find test point that maximizes expected improvement
    #we run L-BFGS-B self.num_restarts times
    def max_EI(self):
        best = {'x':None, 'fun':None}
        for i in range(self.num_restarts):
            result = minimize(
                lambda x: -self.expected_improvement(torch.from_numpy(x).to(dtype=torch.get_default_dtype())),
                x0=self.rand_start_point(),
                method='L-BFGS-B',
                jac=lambda x: -self.ei_grad(torch.from_numpy(x).to(dtype=torch.get_default_dtype())), #provide gradient so it isn't approximated
                bounds=self.bounds
            )
            #store the x value with the best expected improvement over all restarts
            if best['fun'] == None or result['fun'] < best['fun']:
                best['fun'] = result['fun']
                best['x'] = result['x']
                
        return torch.from_numpy(best['x']).to(dtype=torch.get_default_dtype())
    
    #maximal (marginal) likelihood estimation to select optimal parameters for covariance function
    #we run L-BFGS-B self.num_restarts times and pick the covariance parameters that can best output Y, given X
    def mle(self):
        best = {'x':None, 'fun':None}
        for i in range(self.num_restarts):
            result = minimize(
                lambda x: -self.get_log_marginal_likelihood(torch.from_numpy(x).to(dtype=torch.get_default_dtype())),
                x0=self.rand_cov_param(),
                method='L-BFGS-B',
                jac=lambda x: -self.marg_likelihood_grad(torch.from_numpy(x).to(dtype=torch.get_default_dtype())), #manually provide gradient
                bounds=self.cov_param_bounds
            )
            #in this case, best['x'] will store the covariance parameters with the highest log marginal likelihood over all restarts
            if best['fun'] == None or result['fun'] < best['fun']:
                best['fun'] = result['fun']
                best['x'] = result['x']
                
        return torch.from_numpy(best['x']).to(dtype=torch.get_default_dtype())
    
    def step(self):
        #find the x with the highest expected improvement
        point = self.max_EI()
        #compute y value
        out = self.func(point)

        #add new x
        self.Xs = torch.cat([self.Xs, point])
        
        #uncenter Ys
        self.Ys = self.Ys + self.Y_mean
        #add new y
        self.Ys = torch.cat([self.Ys, out])
        #recenter
        self.Y_mean = self.Ys.mean(0, keepdim=True)
        self.Ys = self.Ys - self.Y_mean
        
        #update covariance kernel parameters
        cov_params = self.mle()
        self.length = torch.exp(cov_params[0])
        self.signal_variance = torch.exp(cov_params[1])
        self.noise_variance = torch.exp(cov_params[2])
        
    def train(self):
        #start by updating covariance kernel parameters
        cov_params = self.mle()
        self.length = torch.exp(cov_params[0])
        self.signal_variance = torch.exp(cov_params[1])
        self.noise_variance = torch.exp(cov_params[2])
        
        for i in range(self.num_train_steps):
            self.step()
            print(i, self.Xs[-1], self.Ys[-1])
        max_index = torch.argmax(self.Ys).item()
        return self.Xs[max_index] #return x value of the max of the function

    #this function is almost identical to previous pred_mean_var function
    # but it allows you to choose a subset of all the chosen points of the Gaussian Process
    # so we can display the improvement in predictions over time
    def pred_mean_using_num_points(self, test_x, num_points = None):
        if num_points:
            Xs = self.Xs[:num_points]
            Ys = self.Ys[:num_points]
        else:
            Xs = self.Xs
            Ys = self.Ys
            
        #hard to invert k so we calculate kx = y, x = k^-1 * y, using cholesky decomposition
        # instead of calculating k^-1
        
        cov_fn = self.covariance
        
        k = cov_fn(Xs, Xs)
        jitter = 1e-6 * torch.eye(Xs.shape[0], dtype=k.dtype)
        L = torch.linalg.cholesky(k + self.noise_variance * torch.eye(Xs.shape[0]) + jitter)
        
        #forward substitution
        forward_sub = torch.linalg.solve_triangular(L, Ys.unsqueeze(-1), upper=False)
        #backward substitution
        alpha = torch.linalg.solve_triangular(L.T, forward_sub, upper=True)
        
        k_star = cov_fn(Xs, test_x)
        
        pred_mean = torch.matmul(k_star.T, alpha)
        
        v = torch.linalg.solve_triangular(L, k_star, upper=False)
        pred_var = cov_fn(test_x, test_x) - torch.matmul(v.T,v)
       
        return pred_mean, pred_var



if __name__ == "__main__":
    #create and train Gaussian Process
    gp = Gaussian_Process()
    max_x = gp.train()
    print(max_x)

    #list of positions along x axis
    indices = 2*torch.pi * torch.arange(200)/200
    function_outs = gp.func(indices)

    video_filename = 'output_video.mp4'
    fps = 1
    frames = []

    #we want to display the improvement in the GP predictions as the number of points increases
    # from the 3 space-filling starter points onwards
    for num_points in range(3, 51):
        #given a GP that computes based on the specified number of points, we make predictions across different 
        # positions along the x_axis
        preds = [
            (gp.pred_mean_using_num_points(torch.tensor([i]), num_points=num_points)[0] + gp.Y_mean).item()
            for i in indices
        ]
    
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(indices.numpy(), function_outs.numpy(), label='True') #the actual function
        ax.plot(indices.numpy(), preds, label='Predictions') #gaussian process predictions
        #plot the points this prediction was conditioned on
        ax.plot(gp.Xs.numpy()[:num_points], (gp.Ys + gp.Y_mean).numpy()[:num_points], 'o', label='Points')
    
        ax.set_xlabel("X-axis Label")
        ax.set_ylabel("Y-axis Label")
        ax.set_title(f"Iteration {num_points}")
        ax.legend()
    
        #convert plot to numpy array image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
        frames.append(img)
        plt.close(fig)
    
    #save the frames as an mp4 video
    imageio.mimsave(video_filename, frames, fps=fps)
    print(f"Video saved as {video_filename}")
