from itertools import chain, combinations
from sampling import sampler 
import performance_scores
import numpy as np
import sys
from bundling import kmeans_cluster
import error_measures

''' 
this can be some idea of accuracy of the model. 

input is some subset of the data
'''

def utility_function(data_subset, X, Y, x_test, y_test):
    xcut = X[data_subset, :]
    ycut = Y[data_subset,:]
    s = sampler()
    if np.sum(s.classify(ycut.ravel())) in [len(s.classify(ycut.ravel())),0]:
        return 0
    return performance_scores.logistics_MAE_score(xcut, s.classify(ycut.ravel()), x_test, y_test)

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def banzhaf_bundled_remove_all(subset, utility, N, X, Y, x_test, y_test):
    r = list(range(0, N))
    out_of_bundle = [i for i in r if i not in subset]
    ps = powerset(out_of_bundle)
    total_val = 0.0
    for j in ps:
        if len(j) < 4:
            continue
        without_i = list(j)
        with_i = without_i + subset
        utility_without = utility_function(without_i, X, Y, x_test, y_test)
        utility_with = utility_function(with_i, X, Y, x_test, y_test)
        total_val += (utility_with - utility_without)
    return total_val

'''
N is the length of the dataset
i is the index index into N
'''
def banzhaf(i, utility, N, X, Y, x_test, y_test):
    r = chain(range(0,i) , range(i+1, N))
    ps = powerset(r)
    total_val = 0.0
    for j in ps:
        if len(j) < 4:
            continue
        without_i = list(j)
        with_i = without_i + [i,]
        utility_without = utility_function(without_i, X, Y, x_test, y_test)
        utility_with = utility_function(with_i, X, Y, x_test, y_test)
        total_val += (utility_with - utility_without)
    return total_val



def driver(number_samples, num_clusters):
    N = number_samples
    s = sampler()
    xs = s.sample_X(N)
    #split_x = xs.array_split(2)
    x_train = s.sample_X(N)
    y_train = s.sample_Y(x_train)
    x_test =  s.sample_X(N)
    y_test = s.sample_Y(x_test)
    clusters = kmeans_cluster(x_train, k=num_clusters)
    cluster_map = {}
    for i in range(len(clusters)):
        cluster_map.setdefault(clusters[i],[]).append(i)

    #banzhaf(int(args[1]), utility_function, N, xs, ys)

    bundled_measure =[] 
    for i in range(num_clusters):
        bundled_measure.append(banzhaf_bundled_remove_all(cluster_map[i], 
                                     utility_function, N, x_train, 
                                     y_train, x_test, y_test))
    
    print("======")
    measure = []
    for i in range(N):
        measure.append(banzhaf(i, utility_function, N, x_train, y_train, x_test, y_test))

    print(error_measures.bundle_difference(clusters, np.array(bundled_measure), np.array(measure), error_measures.abs_distance))

    print(error_measures.individual_difference(clusters, np.array(bundled_measure), np.array(measure), error_measures.abs_distance))

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        print("Expected usage:\n python3 banzhaf.py {num_samples} {index to value}")
        sys.exit(0)
    N = int(args[0])
    num_clusters = int(args[1])
    driver(N, num_clusters)


