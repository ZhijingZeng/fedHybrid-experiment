import numpy as np
import pandas as pd

class TrainParam:
    def __init__(self, alpha1 = None, beta1 = None, alpha2 = None, beta2 = None, mu = None, K = None, client_gradient = None, client_Newton = None, initial_x = None):
        self.alpha1 = alpha1 # gradient primal
        self.beta1 = beta1 #gradient dual
        self.alpha2 = alpha2 # newton primal
        self.beta2 = beta2 #newton dual
        self.mu = mu #
        self.K = K # maximum iteration
        self.client_gradient = client_gradient
        self.client_Newton = client_Newton
        self.initial_x = initial_x


# alpha1, beta1, etc are ranges of parameters to be tuned 
class TuneParam:
    def __init__(self, alpha1_range = None, beta1_range = None, alpha2_range = None, beta2_range = None, mu_range = None, K = None, client_gradient = None, client_Newton = None, initial_x = None):
        self.alpha1_range = alpha1_range
        self.beta1_range = beta1_range
        self.alpha2_range = alpha2_range
        self.beta2_range = beta2_range
        self.mu_range = mu_range
        self.K = K
        self.client_gradient = client_gradient # clients doing second-order updates in FedHybrid
        self.client_Newton = client_Newton # clients doing first-order updates in FedHybrid
        self.initial_x = initial_x