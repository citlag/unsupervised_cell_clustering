# Citlalli Gamez Serna

import umap
import json
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder


REDUCER = {
    "PCA": PCA(n_components=50, random_state=0), 
    "UMAP": umap.UMAP(n_components=20, random_state=0),
    "UMAP_GMM": umap.UMAP(n_components=10, random_state=0),
    "UMAP_SPECTRAL": umap.UMAP(n_components=20, n_neighbors=5, random_state=0)
}

CLUSTERER = {
    "KMeans": lambda n_clusters: KMeans(n_clusters=n_clusters, n_init=20, random_state=0),
    "GMM": lambda n_clusters: GaussianMixture(n_components=n_clusters, covariance_type="spherical", random_state=0),
    "Spectral": lambda n_clusters: SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=50, random_state=0)
}

def perform_unsupervised_clustering_with_n_clusters(data, n_clusters, method='KMeans', dim_reduc='UMAP'):
    '''---- Unsupervised clusteting defining the number of clusters ----
    Parameters:
    - data: array with (n,s) where n are the number of objects and s the size of the feature. E.g. (10,1024)
    - n_clusters: (int) number of clusters
    - method: method name to consider for clustering. Valid methods: ['KMeans', 'GMM','Spectral']
    - dim_reduc: method to perform dimensionality reduction. Valid methods: ['PCA', 'UMAP']
    Return:
    - labels: numpy array with the clusters per object with size n (number of objects)
    - probabilities: numpy array of probabilities per object with size (n,m) where n are the number of objects and m the probabilities per cluster
    '''
    probabilities=None
    # ---- Standardize data ----
    data_scaled = StandardScaler().fit_transform(data)
    
    # ---- Perform dimensionality reduction ----
    if method in ['GMM','Spectral'] and dim_reduc=='UMAP': dim_reduc = dim_reduc + '_' + method.upper() 
    reducer = REDUCER[dim_reduc]
    data_reduced = reducer.fit_transform(data_scaled)
    
    # ---- Perform unsupervised clustering ----
	#clusterer = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=0)
    clusterer = CLUSTERER[method](n_clusters)
    labels = clusterer.fit_predict(data_reduced)
    if method == 'GMM':
        probabilities  = clusterer.predict_proba(data_reduced) 
	
    # ---- Evaluate the clusters ----
    score = evaluate_unsupervised_clusters(labels, data_reduced)
    
    return labels, probabilities, score



def evaluate_unsupervised_clusters(labels, embeddings):
    '''---- Evaluate how separable are the clusters with the silhouette score [0,1] ----
    +1.0 → Perfect clustering (very compact, very separated clusters).
    ~0.5–0.7 → Good structure, clusters are meaningful.
    ~0.25–0.5 → Weak structure, some overlap between clusters.
    0 or below → No substantial structure, poor clustering.
    Parameters:
    - labels: numpy array with the clusters per object with size n (number of objects)
    - embeddings: array of size (n,dim) where n are the number of objects and dim are the number of dimensions
    Return:
    - score: float [0,1]
    '''
    # Ignore noisy labels
    mask = labels != -1 if -1 in labels else np.ones_like(labels, dtype=bool)
    if len(np.unique(labels[mask])) > 1:
        score = silhouette_score(embeddings[mask], labels[mask])
    else:
        score = -1  # invalid clustering
    return score



def map_int_labels_to_str_labels(gt_str_labels, pred_int_labels):
    '''---- Map the int cluster ids to the string ground truth labels using the Hungarian algorithm ----
    Works when n_pred == n_true or n_pred > n_true.
    Parameters:
    - gt_str_labels: array with string labels per object with size n (number of objects)
    - pred_int_labels: array wint the int labels of the cluster with size n (number of objects)
    Return:
    - pred_int_labels_mapped: array of predicted objects with size n mapped to the corresponsing int groud truth label
    - pred_str_labels: array of the cluster labels mapped to the string labels of the ground truth (with 'Unmatched' if pred labels are more than the ground truth labels)
    - gt_int_labels: array of ground truth objects converted to int values with size n 
    - mapping_dict: mapping dictionary where keys are the ground truth labels and values are the integer of the predicted cluster
    '''
    
    # Encode string labels to integers
    le = LabelEncoder()
    gt_int_labels = le.fit_transform(gt_str_labels)  # e.g. ["Epithelial", "Neoplastic", "Stromal"] → 0,1,2
    n_gt_classes = len(le.classes_)
    n_pred_classes = len(np.unique(pred_int_labels))
    
    # Compute confusion matrix between true ints and predicted ints
    cm = confusion_matrix(gt_int_labels, pred_int_labels)
    
    # Hungarian algorithm to optimally align clusters 
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize agreement
    
    # Check if ground truth classes are less than the predicted ones
    if n_gt_classes < n_pred_classes: 
        # Build mapping: cluster_id -> true_label_int (for mapped clusters)
        mapping = {col: row for row, col in zip(row_ind, col_ind) if row < len(le.classes_)}
        # Remap predicted labels to aligned labels, unknown clusters -> -1
        pred_int_labels_mapped = np.array([mapping.get(c, -1) for c in pred_int_labels])
        # Confusion matrix after alignment
        cm_aligned = confusion_matrix(gt_int_labels, pred_int_labels_mapped, labels=[0,1,2,3,-1])
        # Recover string labels for aligned predictions
        pred_str_labels = np.array([le.inverse_transform([c])[0] if c != -1 else "Unmatched"
                           for c in pred_int_labels_mapped])
    else: 
        mapping = dict(zip(col_ind, row_ind))
        pred_int_labels_mapped = np.array([mapping[c] for c in pred_int_labels])
        cm_aligned = confusion_matrix(gt_int_labels, pred_int_labels_mapped)
        pred_str_labels = le.inverse_transform(pred_int_labels_mapped)
    
    #print("Original confusion matrix:\n", cm)
    #print("Aligned confusion matrix:\n", cm_aligned)
    
    mapping_dict = {cls: idx for idx, cls in enumerate(le.classes_)}
    if 'Unmatched' in pred_str_labels: 
        mapping_dict['Unmatched']=n_pred_classes-1
        pred_int_labels_mapped[pred_int_labels_mapped==-1]=n_pred_classes-1
    
    # Return
    return pred_int_labels_mapped, pred_str_labels, gt_int_labels, mapping_dict
    
    

def evaluate_gt_vs_unsupervised_clusters(y_true_int,y_pred_int):
    '''---- Evaluate ground truth labels vs the predicted ones with: ----
    Confusion matrix
    ARI: Adjusted Rand Index measures how similar the prediction is to the ground truth [0=random labeling, 1=perfect match]
    NMI: Normalized Mutual Information measures shared information between cluster assigment and ground truth [0=no relation, 1=perfect match]
    Parameters:
    - y_true_int: array with int ground truth labels with size n (number of objects)
    - y_pred_int: array wint int predicted labels with size n (number of objects)
    Return:
    A dictionary with the following metrics:
    - cm: confusion matrix
    - ari: float value of the adjusted rand index
    - nmi: float value of the normalized mutual information 
    '''
    cm = confusion_matrix(y_true_int, y_pred_int)
    ari = adjusted_rand_score(y_true_int, y_pred_int)
    nmi = normalized_mutual_info_score(y_true_int, y_pred_int)
    
    print("Confusion matrix:\n", cm)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    return {'Confusion matrix':cm, 'ARI':ari, 'NMI':nmi}



def make_geojson_feature(cell_id, coords, cluster_id, properties_dict):
    ''' ---- Make a Feature per cell to store it as a collection ---- 
    - cell_id: int cell id
    - coords: list of lists with x,y coordiantes [[1,2],[2,4],...]
    - cluster_id: cluster id
    - properties_dict: dictionary with keys ['magnification','cell_model','model_version','slide_name']
    Return
    - Feature structure for geojson file
    '''
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": coords
        },
        "properties": {
            "id": int(cell_id),
            "model_label": 'Cluster ' + str(cluster_id),
            "model_class_id": int(cluster_id),
            "model_magnification": properties_dict['magnification'],
            "class_type": "object",
            "model_name": properties_dict['cell_model'],
            "model_version": properties_dict['model_version'],
            "anet_class_label": 'Cluster ' + str(cluster_id),
            "slide": properties_dict['slide_name']
        }
    }


def save_cell_info_geoson(path_out, ids, coords, labels, properties_dict ):
    ''' ---- save the cell information of the unsupervised clusters in a geojson file ---- 
    - path_out: output path directory
    - ids: numpy array of n cell ids
    - coords: numpy array of n number of cells where each row has its numpy array of (m,2) points
    - labels: numpy array with the cluster id per cell
    - properties_dict: dictionary with keys ['magnification','cell_model','model_version','slide_name']
    '''
    features = []
    for i in range(len(ids)): 
        features.append(make_geojson_feature(ids[i],coords[i],labels[i], properties_dict))
    
    feature_collection = {'type':'FeatureCollection', 'features':features}
    
    with open(path_out, 'w') as f:
        json.dump(feature_collection, f, indent=2)
    
    print('[INFO] Geojson file saved at: ', path_out)


def save_cell_info_geoson_copy(path_out, ids, coords, labels, properties_dict ):
    ''' ---- save the cell information of the unsupervised clusters in a geojson file ---- 
    - path_out: output path directory
    - ids: numpy array of n cell ids
    - coords: numpy array of n number of cells where each row has its numpy array of (m,2) points
    - labels: numpy array with the cluster id per cell
    - properties_dict: dictionary with keys ['magnification','cell_model','model_version','slide_name']
    '''
    features = []
    for i in range(len(ids)): 
        features.append(make_geojson_feature(ids[i],coords[i].astype(int).tolist(),labels[i], properties_dict))
    
    feature_collection = {'type':'FeatureCollection', 'features':features}
    
    with open(path_out, 'w') as f:
        json.dump(feature_collection, f, indent=2)
    
    print('[INFO] Geojson file saved at: ', path_out)