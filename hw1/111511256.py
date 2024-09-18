import numpy as np

def initialize_centroids_kmeans_plus_plus(X, k):
    """
    Initialize k centroids using the K-means++ method.
    """
    n_points = X.shape[0]
    centroids = np.zeros((k, X.shape[1]))
    
    # Randomly choose the first centroid from the data points
    centroids[0] = X[np.random.choice(n_points)]
    
    # Compute the remaining k-1 centroids
    for i in range(1, k):
        distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids[i] = X[j]
                break
    
    return centroids

def compute_distances(X, centroids):
    """
    Compute the distance from each point in X to each centroid.
    """
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

def update_centroids(X, labels, k):
    """
    Update the centroids as the mean of all points assigned to each centroid.
    """
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        if points.shape[0] > 0:
            centroids[i] = points.mean(axis=0)
    return centroids

def kmeans_clustering(X, k, max_iters=100):
    """
    K-means clustering algorithm using numpy.
    """
    centroids = initialize_centroids_kmeans_plus_plus(X, k)
    for _ in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels

if __name__ == "__main__":
    # load data
    X = np.load("./data.npy") # size: [10000, 512]

    # Choose the number of clusters
    n_clusters = 13

    y = kmeans_clustering(X, k=n_clusters)

    # save clustered labels
    np.save("111511256.npy", y) # output size should be [10000]
    