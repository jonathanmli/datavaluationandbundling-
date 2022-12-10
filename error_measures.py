import numpy as np

def squared_distance(x,y):
    return (x-y)**2

def abs_distance(x,y):
    return abs(x-y)

'''
calculates the bundle difference as defined in the paper

partition: vector of parititions
bundled_measure[j]: vector mapping from bundle index j to value
measure[i]: vector mapping from data index i to value
distance: function for distance metric

note that partition and measure should have same dim n, bundled_measure should have dim k
'''
def bundle_difference(partition, bundled_measure, measure, distance):
    k = len(bundled_measure)
    n = len(partition)
    if n != len(measure):
        raise Exception()
    
    true_bundled_measure = np.zeros_like(bundled_measure)
    for j in range(k):
        true_bundled_measure[j] = np.sum(measure[partition == j])

    return np.sum(distance(true_bundled_measure, bundled_measure))

'''
calculates the individual difference as defined in the paper

partition: vector of parititions
bundled_measure[j]: vector mapping from bundle index j to value
measure[i]: vector mapping from data index i to value
distance: function for distance metric

note that partition and measure should have same dim n, bundled_measure should have dim k
'''
def individual_difference(partition, bundled_measure, measure, distance):
    k = len(bundled_measure)
    n = len(partition)
    if n != len(measure):
        raise Exception()

    _, bundle_sizes = np.unique(partition, return_counts=True)
    bundled_measures = bundled_measure/bundle_sizes
    
    ind_bund_measure = np.zeros_like(measure)
    for j in range(k):
        ind_bund_measure[partition==j] = bundled_measures[j]
    # for i in range(n):
    #     ind_bund_measure[i] = bundled_measures[partition[i]]
    return np.sum(distance(measure, ind_bund_measure))