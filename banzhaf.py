from itertools import chain, combinations

''' 
this can be some idea of accuracy of the model. 

input is some subset of the data

- hardware assisted data valuation 
'''
def utility_function(data_subset):
    return 1

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

'''
N is the length of the dataset
i is the index index into N
'''
def banzhaf(i, utility, N) :
    r = chain(range(0,i) , range(i+1, N))
    ps = powerset(r)
    total_val = 0.0
    for j in ps:
        without_i = list(j)
        with_i = without_i + [i,]
        utility_without = utility_function(without_i)
        utility_with = utility_function(with_i)
        total_val += (utility_with - utility_without)
    print(total_val)


banzhaf(5, utility_function, 10)
