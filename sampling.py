import numpy as np

class sampler:

    # note that first beta is intercept
    def __init__(self, eps_sd = 0.1, betas = np.array([0,1,-1]), Xmean = np.array([0,0]), Xcov = None) -> None:
        np.random.seed(0)
        self.Xcov = Xcov
        self.Xmean = Xmean 
        self.betas = betas
        if self.Xcov is None:
            self.Xcov = np.identity(len(self.Xmean))
        self.eps_sd = eps_sd

    def _logistics_f(self, x):
        return 1/(1+np.exp(-x))
    
    # samples n predictors X from Gaussian distribution, with constant column
    def sample_X(self, n):
        return np.concatenate((np.ones((n,1)), np.random.multivariate_normal(self.Xmean, self.Xcov, n)), axis=1)
    
    # samples predictions Y from predictors X with noise
    def sample_Y(self, X):
        eps = np.random.normal(0, self.eps_sd,len(X))
        return np.expand_dims(self._logistics_f(np.matmul(X, self.betas) + eps), axis=1)

    # returns the true Y values given X
    def compute_Y(self, X):
        return self._logistics_f(np.matmul(X, self.betas))
    
    # samples from joint X, Y distribution
    def sample_XY(self, n):
        X = self.sample_X(n)
        Y = self.sample_Y(X)
        print(X.shape)
        print(Y.shape)
        return np.concatenate((X, Y), axis=1)

    def classify(self, Y):
        return Y > 0.5
