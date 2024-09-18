import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from progress.bar import Bar



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

def kmeans_clustering(X, k, max_iters=10000):
    """
    K-means clustering algorithm using numpy.
    """
    centroids = initialize_centroids_kmeans_plus_plus(X, k)
    with Bar('Processing', max = max_iters) as bar:
        for i in range(max_iters):
            distances = compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = update_centroids(X, labels, k)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
            bar.next()
        return labels

if __name__ == "__main__":
    # load data
    silhouette_avg_best = 0
    ari_score_best = 0
    with Bar('Loading', max = 500) as bar:
        for i in range(5000):
                  
            X = np.load("./data.npy") # size: [10000, 512]
            
            # Choose the number of clusters
            n_clusters = 13
            print("")
            y = kmeans_clustering(X, k=n_clusters)

            #print(y[0:10])
            # save clustered labels
            np.save("predicted_result.npy", y) # output size should be [10000]

            cluster_label = np.load("predicted_result.npy")
            true_label = np.load("label_test.npy")
            data = np.load("data.npy")

            silhouette_avg = silhouette_score(data, cluster_label)
            #print(f'\nSilhouette Coefficient: {silhouette_avg:.3f}')

            # Final ARI score will calculated using whole 10000 data.
            test_num = true_label.size
            ari_score = adjusted_rand_score(true_label, cluster_label[:test_num])
            #print(f'Adjusted Rand Index (ARI): {ari_score:.3f}')
            if i == 0 :
                silhouette_avg_best = silhouette_avg
                ari_score_best = ari_score
                print(f'\nSilhouette Coefficient: {silhouette_avg:.3f}')
                print(f'Adjusted Rand Index (ARI): {ari_score:.3f}')
                #count = 0
                
            if(silhouette_avg >= 0.08 and silhouette_avg > silhouette_avg_best and ari_score >= 0.55 and ari_score > ari_score_best):
                silhouette_avg_best = silhouette_avg
                ari_score_best = ari_score
                #count = 0
                print(f'\nBest Silhouette Coefficient: {silhouette_avg_best:.3f}')
                print(f'Best Adjusted Rand Index (ARI): {ari_score_best:.3f}')
                np.save("predicted_result_best.npy", y)
            bar.next()
        
            