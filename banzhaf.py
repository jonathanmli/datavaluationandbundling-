from itertools import chain, combinations
from sampling import sampler 
import performance_scores
import numpy as np
import sys

''' 
this can be some idea of accuracy of the model. 

input is some subset of the data
'''

def utility_function(data_subset, X, Y):
    xcut = X[data_subset, :]
    ycut = Y[data_subset,:]
    if np.sum(s.classify(ycut.ravel())) in [len(s.classify(ycut.ravel())),0]:
        return 0
    return performance_scores.logistics_MAE_score(xcut, s.classify(ycut.ravel()), xcut, ycut)

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

'''
N is the length of the dataset
i is the index index into N
'''
def banzhaf(i, utility, N, X, Y):
    r = chain(range(0,i) , range(i+1, N))
    ps = powerset(r)
    total_val = 0.0
    for j in ps:
        if len(j) < 4:
            continue
        without_i = list(j)
        with_i = without_i + [i,]
        utility_without = utility_function(without_i, X, Y)
        utility_with = utility_function(with_i, X, Y)
        total_val += (utility_with - utility_without)
    print(total_val)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print("Expected usage:\n python3 banzhaf.py {num_samples} {index to value}")
        sys.exit(0)
    N = int(args[0])
    s = sampler()
    xs = s.sample_X(N)
    ys = s.sample_Y(xs)
    banzhaf(int(args[1]), utility_function, N, xs, ys)
