# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # file_path = '/content/drive/MyDrive/BreastCancer/data/allGeneExprs.csv'  
# file_path = '/content/drive/MyDrive/BreastCancer/data/Mu_all_BTFs.csv'  

# data = pd.read_csv(file_path)

# gene_ids = data.iloc[:, 0]
# gene_expression_data = data.iloc[:, 1:].values

# # print(gene_ids)
# # print(gene_expression_data)

# # Number of clusters (choose based on your specific needs)
# n_clusters = 3

# # Perform k-means clustering
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# clusters = kmeans.fit_predict(gene_expression_data)

# # Print cluster assignments
# # for gene_id, cluster_id in zip(gene_ids, clusters):
# #     print(f"Gene {gene_id} is in cluster {cluster_id}")

# # Visualize the clusters using PCA
# pca = PCA() #n_components=2
# reduced_data = pca.fit_transform(gene_expression_data)
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('K-means Clustering of Gene Expression Data')
# plt.colorbar(label='Cluster ID')

# # plt.show()

# output_file_path = 'revision/multi_clusters.png' 
# plt.savefig(output_file_path)
# plt.close()





# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Load gene expression data from CSV file
# # file_path = 'path_to_your_file.csv' 
# # data = pd.read_csv(file_path)


# gene_ids = data.iloc[:, 0]
# gene_expression_data = data.iloc[:, 1:].values

# silhouette_scores = []
# k_values = range(2, 10)
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     clusters = kmeans.fit_predict(gene_expression_data)
#     score = silhouette_score(gene_expression_data, clusters)
#     silhouette_scores.append(score)
#     print(f'K-means with k={k}, silhouette score={score:.3f}')

# plt.plot(k_values, silhouette_scores, marker='o')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Scores for K-means')
# # plt.show()
# output_file_path = 'revision/kmean.png'
# plt.savefig(output_file_path)

# silhouette_scores_dbscan = []
# eps_values = np.linspace(0.1, 10.0, 10)
# for eps in eps_values:
#     dbscan = DBSCAN(eps=eps, min_samples=5)
#     clusters = dbscan.fit_predict(gene_expression_data)
#     if len(set(clusters)) > 1:  
#         score = silhouette_score(gene_expression_data, clusters)
#         silhouette_scores_dbscan.append(score)
#         print(f'DBSCAN with eps={eps:.1f}, silhouette score={score:.3f}')
#     else:
#         silhouette_scores_dbscan.append(-1)

# # Plot silhouette scores for DBSCAN
# plt.plot(eps_values, silhouette_scores_dbscan, marker='o')
# plt.xlabel('eps value')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Scores for DBSCAN')
# output_file_path = 'revision/DBSCAN.png'  
# plt.savefig(output_file_path)


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.cm as cm  # Import colormap

file_path = '/content/drive/MyDrive/BreastCancer/data/Mu_all_BTFs.csv' 
# file_path = '/content/drive/MyDrive/BreastCancer/data/allGeneExprs.csv'  
data = pd.read_csv(file_path)

gene_ids = data.iloc[:, 0]
gene_list = gene_ids.to_list()
gene_expression_data = data.iloc[:, 1:].values
print(len(gene_expression_data))

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(gene_expression_data)

cluster_centers = kmeans.cluster_centers_
representative_genes = []

for i in range(n_clusters):
    distances = cdist([cluster_centers[i]], gene_expression_data, 'euclidean')[0]
    closest_index = np.argmin(distances)
    representative_genes.append((gene_ids[closest_index], gene_expression_data[closest_index]))

for cluster_id, (gene_id, expression) in enumerate(representative_genes):
    print(f"Cluster {cluster_id} representative gene: {gene_id}")
    # print(f"Gene expression: {expression}")


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(gene_expression_data)
# plt.figure(figsize=(6, 4),dpi=300)
plt.style.use('seaborn-darkgrid')

cmap = cm.get_cmap('viridis')
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap=cmap, alpha=0.7, s=70)
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, alpha=1)#cmap='viridis', 

for i, (gene_id, expression) in enumerate(representative_genes):
    centroid_pca = pca.transform([expression])
    plt.scatter(centroid_pca[0][0], centroid_pca[0][1], c='red', marker='x', s=70)
    plt.text(centroid_pca[0][0], centroid_pca[0][1], f'{gene_id}'.replace('TCGA.',''), fontsize=13, color='red', weight='bold')

plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
# plt.title('K-means Clustering of Multi-Omic Data')
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()


output_file_path = 'revision/clusterResults/final_multi_.png'  
plt.savefig(output_file_path, dpi=1000)
plt.close()

print('---------------------------------')

# file_path = '/content/drive/MyDrive/BreastCancer/data/Mu_all_BTFs.csv'
file_path = '/content/drive/MyDrive/BreastCancer/data/allGeneExprs.csv'  
data = pd.read_csv(file_path)

# gene_ids = data.iloc[:, 0]
# gene_expression_data = data.iloc[:, 1:].values
data = data[data['ID'].isin(gene_list)]
gene_ids = data.iloc[:, 0]
gene_expression_data = data.iloc[:, 1:].values
print(len(gene_expression_data))

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(gene_expression_data)

cluster_centers = kmeans.cluster_centers_
representative_genes = []

for i in range(n_clusters):
    distances = cdist([cluster_centers[i]], gene_expression_data, 'euclidean')[0]
    closest_index = np.argmin(distances)
    representative_genes.append((gene_ids.iloc[closest_index], gene_expression_data[closest_index]))

for cluster_id, (gene_id, expression) in enumerate(representative_genes):
    print(f"Cluster {cluster_id} representative gene: {gene_id}")
    # print(f"Gene expression: {expression}")

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(gene_expression_data)
plt.style.use('seaborn-darkgrid')

cmap = cm.get_cmap('viridis')
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap=cmap, alpha=0.7, s=70)

# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, alpha=1)
for i, (gene_id, expression) in enumerate(representative_genes):
    centroid_pca = pca.transform([expression])
    plt.scatter(centroid_pca[0][0], centroid_pca[0][1], c='red', marker='x', s=70)
    plt.text(centroid_pca[0][0], centroid_pca[0][1], f'{gene_id}'.replace('TCGA.',''), fontsize=13, color='red', weight='bold')

plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
# plt.title('K-means Clustering of Gene Expression Data')
plt.colorbar(scatter,label='Cluster ID')
plt.tight_layout()
# plt.style.use('seaborn-darkgrid')
# plt.show()

output_file_path = 'revision/clusterResults/final_gene_.png'  
plt.savefig(output_file_path, dpi=1000)
plt.close()

# print(f"Cluster plot saved as {output_file_path}")












# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt
# from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import pdist
# from sklearn.decomposition import PCA

# def read_csv(file_path):
#     data = pd.read_csv(file_path)
#     ids = data.iloc[:, 0]
#     features = data.iloc[:, 1:]
#     return ids, features

# # K-means clustering
# def kmeans_clustering(features, n_clusters):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(features)
#     return clusters, kmeans.inertia_

# # K-means++ clustering
# def kmeans_pp_clustering(features, n_clusters):
#     kmeans_pp = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
#     clusters = kmeans_pp.fit_predict(features)
#     return clusters, kmeans_pp.inertia_

# # Silhouette Coefficient
# def silhouette_coefficient(features, labels):
#     return silhouette_score(features, labels)

# # Dunn's Index calculation
# def dunn_index(features, labels):
#     distances = pairwise_distances(features)
#     unique_labels = np.unique(labels)
#     inter_cluster_distances = []
#     intra_cluster_distances = []
    
#     for label in unique_labels:
#         cluster_points = features[labels == label]
#         intra_cluster_distances.append(np.mean(cdist(cluster_points, cluster_points, 'euclidean')))
#         for other_label in unique_labels:
#             if label != other_label:
#                 other_cluster_points = features[labels == other_label]
#                 inter_cluster_distances.append(np.mean(cdist(cluster_points, other_cluster_points, 'euclidean')))
                
#     dunn = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
#     return dunn

# # Determine best number of clusters using Silhouette Coefficient and Dunn's Index
# def determine_best_clusters(features, max_clusters=10):
#     silhouette_scores = []
#     dunn_indices = []
#     inertias = []
    
#     for n_clusters in range(2, max_clusters + 1):
#         clusters, inertia = kmeans_pp_clustering(features, n_clusters)
#         silhouette_scores.append(silhouette_coefficient(features, clusters))
#         dunn_indices.append(dunn_index(features, clusters))
#         inertias.append(inertia)
        
#     best_silhouette_clusters = np.argmax(silhouette_scores) + 2
#     best_dunn_clusters = np.argmax(dunn_indices) + 2
    
#     print(f"Best number of clusters (Silhouette Coefficient): {best_silhouette_clusters}")
#     print(f"Best number of clusters (Dunn's Index): {best_dunn_clusters}")
    
#     return best_silhouette_clusters, best_dunn_clusters, silhouette_scores, dunn_indices, inertias

# # Plot clustered diagrams
# def plot_clusters(features, clusters, title):
#     pca = PCA(n_components=2)
#     reduced_data = pca.fit_transform(features)
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
#     # plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=clusters, cmap='viridis')
#     plt.title(title)
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     # plt.show()
#     plt.savefig(f'revision/{title}.png')

# # Main function
# def main(file_path, name=''):
#     ids, features = read_csv(file_path)
#     best_silhouette_clusters, best_dunn_clusters, silhouette_scores, dunn_indices, inertias = determine_best_clusters(features)
    
#     best_clusters = max(best_silhouette_clusters, best_dunn_clusters, key=lambda x: (silhouette_scores[x-2], dunn_indices[x-2]))
#     # clusters, _ = kmeans_pp_clustering(features, best_clusters)
#     clusters, _ = kmeans_clustering(features, best_clusters)
    
#     plot_clusters(features, clusters, name+f"Best Clusters (n={best_clusters})")


# file_path = '/content/drive/MyDrive/BreastCancer/data/Mu_all_BTFs.csv' 
# # file_path = '/content/drive/MyDrive/BreastCancer/data/allGeneExprs.csv' 
# main(file_path, 'gene exp kmean')

















