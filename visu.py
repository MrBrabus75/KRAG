# visu.py
import faiss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_faiss_index():
    return faiss.read_index("faiss_index.index")

def visualize_index_stats():
    index = load_faiss_index()
    
    # Afficher le nombre de vecteurs dans l'index
    print(f"Number of vectors in the index: {index.ntotal}")

    # Extraire les vecteurs de l'index
    vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    for i in range(index.ntotal):
        index.reconstruct(i, vectors[i])

    # Calculer les distances moyennes entre les vecteurs
    distances = []
    for i in range(index.ntotal):
        for j in range(i + 1, index.ntotal):
            distances.append(np.linalg.norm(vectors[i] - vectors[j]))
    
    # Afficher les statistiques des distances
    print(f"Mean distance between vectors: {np.mean(distances)}")
    print(f"Median distance between vectors: {np.median(distances)}")
    print(f"Standard deviation of distances: {np.std(distances)}")
    
    # Visualisation des distances
    plt.figure(figsize=(12, 6))
    sns.histplot(distances, kde=True, bins=50)
    plt.title('Distribution of Distances Between Vectors')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()

    # Visualisation des embeddings en 2D avec PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(12, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
    plt.title('2D PCA Projection of Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    
    # Afficher les composantes principales
    print(f"Explained variance by first component: {pca.explained_variance_ratio_[0]:.2f}")
    print(f"Explained variance by second component: {pca.explained_variance_ratio_[1]:.2f}")

    # Visualisation des embeddings en 3D avec PCA
    from mpl_toolkits.mplot3d import Axes3D
    pca_3d = PCA(n_components=3)
    vectors_3d = pca_3d.fit_transform(vectors)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2], alpha=0.6)
    ax.set_title('3D PCA Projection of Embeddings')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.show()

    # Afficher les composantes principales en 3D
    print(f"Explained variance by first component: {pca_3d.explained_variance_ratio_[0]:.2f}")
    print(f"Explained variance by second component: {pca_3d.explained_variance_ratio_[1]:.2f}")
    print(f"Explained variance by third component: {pca_3d.explained_variance_ratio_[2]:.2f}")

if __name__ == "__main__":
    visualize_index_stats()
