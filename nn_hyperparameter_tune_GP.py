import math
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import math
from scipy.optimize import minimize

((all_train_data, all_train_labels), (test_data, test_labels)) = mnist.load_data()

unique_classes, counts = np.unique(all_train_labels, return_counts=True)
num_labels = unique_classes.shape[0]

#shuffle and use the last 10000 datapoints as the validation set
indices = np.arange(all_train_data.shape[0])
np.random.shuffle(indices)
val_data = all_train_data[indices[50000:]]
val_labels = all_train_labels[indices[50000:]]
train_data = all_train_data[indices[:50000]]
train_labels = all_train_labels[indices[:50000]]

#normalize
train_data = (train_data/255).astype('float32')
val_data = (val_data/255).astype('float32')
test_data = (test_data/255).astype('float32')

#flatten image data
train_data = train_data.reshape(train_data.shape[0], -1)
val_data = val_data.reshape(val_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)

#function that intializes and returns models
def build_model(num_layers=1, 
                num_hidden_units=16,
                lr=0.001,
                l1_reg=1e-8,
                l2_reg=1e-8,
                dropout_rate=0.0,
                hidden_layer_decrease=0.0 #percentage of hidden unit decrease for each subsequent layer
               ):
    current_hidden_units = num_hidden_units
    
    model = models.Sequential()
    model.add(layers.Input((train_data.shape[-1],)))
    for i in range(num_layers):
        model.add(layers.Dense(
            current_hidden_units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        ))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
        
        #decrease number of hidden units for each subsequent layer, by hidden_layer_decrease
        # we keep a minimum of num_labels hidden units (10)
        current_hidden_units = max(round(current_hidden_units*(1-hidden_layer_decrease)), num_labels)
            
    model.add(layers.Dense(num_labels, activation='softmax'))
                 
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5, #if we don't reach a new minimum loss within 5 epochs we end training
    verbose=0,
    mode='min',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
)

class Gaussian_Process:
    def __init__(self):
        torch.set_default_dtype(torch.float64)

        self.Xs = torch.tensor([]) #list of hyperparameter configurations
        self.Ys = torch.tensor([]) #list of mean-centered resulting accuracies of the above hyperparameter configurations
        self.Y_mean = torch.tensor(0.0) #mean accuracy
        
        #radial basis function covariance
        #for further detials, see: https://www.cs.toronto.edu/~duvenaud/cookbook/
        #we'll initialize noise variance and signal variance once we have the initial space-filling points
        self.noise_variance = None
        self.signal_variance = None
        self.length = 1 #we'll start at 1
        #we unsqueeze x1 and x2 at the first and zeroth dimension, respectively, since we want 
        # pairwise squared distances, across all possible pairs of x1 and x2
        self.covariance = lambda x1, x2: self.signal_variance * torch.exp(
            -0.5*torch.sum((x1.unsqueeze(1)-x2.unsqueeze(0))**2, dim=-1)/(self.length**2)
        )
        
        #the bounds the each hyperparameter must fall in
        self.bounds = [
            (math.log(1e-4), math.log(1e-2)), #lr
            (0.0, 0.5),                       #dropout
            (math.log(1e-8), math.log(1e-2)), #l1
            (math.log(1e-8), math.log(1e-2)), #l2
            (1.0, 7.0),                       #num_layers
            (math.log(2**5), math.log(2**11)),#num_hidden_units
            (0.0,0.2)                         #hidden_units_decrease
        ]
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
        
        #we run the l-bfgs-b algorithm with 30 different random starting points
        # since l-bfgs-b is susceptible to local optima
        self.num_restarts = 50
        
        self.num_train_steps = 150
        
        self.num_epochs_per_run = []
        
    def func(self, x):
        #the gaussian process deals with many of the hyperparameters in a log scale
        # so we use torch.exp to get the real values
        learning_rate = torch.exp(x[0]).item()
        dropout_rate = x[1].item()
        l1_reg = torch.exp(x[2]).item()
        l2_reg = torch.exp(x[3]).item()
        num_layers = int(torch.round(x[4]).item())
        units_per_layer = int(torch.round(torch.exp(x[5])).item())
        hidden_unit_decrease = x[6].item()
        
        print("Run: ", self.Xs.shape[0])
        print("learning_rate: ", learning_rate, 
              "dropout_rate: ", dropout_rate, 
              "l1_reg: ", l1_reg, 
              "l2_reg: ", l2_reg, 
              "num_layers: ", num_layers, 
              "units_per_layer: ", units_per_layer, 
              "hidden_unit_decrease: ", hidden_unit_decrease)
        
        #build model and train
        model = build_model(num_layers=num_layers,
                    num_hidden_units=units_per_layer,
                    lr=learning_rate,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg, 
                    dropout_rate=dropout_rate,
                    hidden_layer_decrease=hidden_unit_decrease)

        history = model.fit(train_data,
                            train_labels,
                            epochs=200,
                            batch_size=512,
                            validation_data=(val_data, val_labels),
                            shuffle=False, 
                            callbacks=[callback],
                            verbose = 0
                           )
        print("Trained for epochs: ", len(history.history['val_accuracy']),
              ' Reached accuracy: ', max(history.history['val_accuracy']))
        print()
        
        #store the number of epochs the model was trained for
        self.num_epochs_per_run.append(len(history.history['val_accuracy']))
        
        return max(history.history['val_accuracy'])
        
    def get_pred_mean_var(self, test_x):
        #implements part of Algorithm 2.1 of the Gaussian Process for Machine learning book
        #Computing: P(y_*|test_x,X,Y)
        
        #hard to invert k so we calculate kx = y, x = k^-1 * y, using cholesky decomposition
        # instead of calculating k^-1
        
        test_x = test_x.unsqueeze(0) #new shape -> [batch_size=1, input_size=7]
        
        cov_fn = self.covariance
        
        k = cov_fn(self.Xs, self.Xs)
        jitter = 1e-6 * torch.eye(self.Xs.shape[0], dtype=k.dtype) #we add this for numerical stability
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
        
        #cov_params should be in log scale at this point
        
        cov_params = torch.exp(cov_params)
        length = cov_params[0]
        signal_variance = cov_params[1]
        noise_variance = cov_params[2]
        
        #new covariance function using the provided cov_params as opposed to the global params
        cov_fn = lambda x1, x2: signal_variance * torch.exp(
            -0.5*torch.sum((x1.unsqueeze(1)-x2.unsqueeze(0))**2, dim=-1)/(length**2)
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
    
    #get random starting hyperparameter configuration within bounds
    def rand_start_point(self):
        point = self.lb + (self.ub - self.lb) * torch.rand(len(self.bounds))
        return point
    
    #get random covariance function parameters within the provided bounds
    def rand_cov_param(self):
        point = self.cov_param_lb + (self.cov_param_ub - self.cov_param_lb) * torch.rand(len(self.cov_param_bounds))
        return point
    
    #find test point that maximizes expected improvement
    #we run L-BFGS-B self.num_restarts times and pick the hyperparameters with the best accuracy
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
                jac=lambda x: -self.marg_likelihood_grad(torch.from_numpy(x).to(dtype=torch.get_default_dtype())), #provide gradient
                bounds=self.cov_param_bounds
            )
            if best['fun'] == None or result['fun'] < best['fun']:
                best['fun'] = result['fun']
                best['x'] = result['x']
                
        return torch.from_numpy(best['x']).to(dtype=torch.get_default_dtype())
    
    def step(self):
        #find the hyperparameter with the highest expected improvement
        point = self.max_EI()
        #compute the accuracy given these hyperparametrs
        out = self.func(point)
        
        #add hyperparameters to list
        self.Xs = torch.cat([self.Xs, point.unsqueeze(0)])
        
        #uncenter Ys
        self.Ys = self.Ys + self.Y_mean
        #add new y
        self.Ys = torch.cat([self.Ys, torch.tensor([out])])
        #recenter
        self.Y_mean = self.Ys.mean(0, keepdim=True)
        self.Ys = self.Ys - self.Y_mean
        
        #update covariance kernel parameters
        cov_params = self.mle()
        self.length = torch.exp(cov_params[0])
        self.signal_variance = torch.exp(cov_params[1])
        self.noise_variance = torch.exp(cov_params[2])
        
    def train(self):
        #get 3 random space-filling points
        if self.Xs.shape[0] == 0:
            for i in range(3):
                point = gp.rand_start_point()
                out = gp.func(point)
                self.Xs = torch.cat([self.Xs, point.unsqueeze(0)])

                self.Ys = self.Ys + self.Y_mean
                self.Ys = torch.cat([self.Ys, torch.tensor([out])])
                self.Y_mean = self.Ys.mean(0, keepdim=True)
                self.Ys = self.Ys - self.Y_mean
            
            #initialize noise_variance and signal_variance
            self.noise_variance = 0.5 * torch.var(self.Ys) #start at 0.5 of variance of y
            self.signal_variance = 0.5 * torch.var(self.Ys) #start at 0.5 of variance of y
            
            print("------ Obtained Initial Random Points. Commencing Training ... ------")
            print()
        
        #start by updating covariance kernel parameters
        cov_params = self.mle()
        self.length = torch.exp(cov_params[0])
        self.signal_variance = torch.exp(cov_params[1])
        self.noise_variance = torch.exp(cov_params[2])
        
        #keep finding the best probable hyperparameters and evaluating them
        for i in range(self.num_train_steps):
            self.step()
            print(i, self.Xs[-1], self.Ys[-1])
        max_index = torch.argmax(self.Ys).item()
        return self.Xs[max_index]


if __name__ == "__main__":
    gp = Gaussian_Process()
    gp.train()

    #save GP params
    torch.save({"Xs": gp.Xs,
                "Ys": gp.Ys,
                "Y_mean": gp.Y_mean,
                "length": gp.length,
                "signal_var": gp.signal_variance,
                "noise_var" :gp.noise_variance
               }, "mnist_hyper_tune_gaussian_process.pt")
