from sklearn.cluster import KMeans

'''
returns partition function after clustering
'''
def kmeans_cluster(X, k=3):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return kmeans.labels_



