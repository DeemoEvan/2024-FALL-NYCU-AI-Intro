import numpy as np
import matplotlib.pyplot as plt

def k_distance_plot(X, k):
    """
    Plot k-distance graph to help determine the optimal eps value.
    """
    from sklearn.neighbors import NearestNeighbors

    dataset = X
    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

if __name__ == "__main__":
    # load data
    X = np.load("./data.npy") # size: [10000, 512]

    # Plot k-distance graph
    k_distance_plot(X, k=5)
    y = np.load("predicted_result.npy")
    z = np.load("label_test.npy") 
    print(y[0:10])
    print(y[10000:9990])
    print(z[0:10])