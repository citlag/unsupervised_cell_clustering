# Citlalli Gamez Serna

import os
import numpy as np
from lib.arg_parser_yaml import get_args
from lib.annotation_utils import read_cell_annotation_file, save_cell_annotation_file
from lib.hibouL_features import extract_features_hibouL
from lib.unsupervised_clustering import perform_unsupervised_clustering_with_n_clusters, map_int_labels_to_str_labels
from lib.unsupervised_clustering import evaluate_gt_vs_unsupervised_clusters, save_cell_info_geoson


def extract_cell_features(args, save_file=True ):
    ''' ---- Extract cell embeddings with Hibou-L model reading the cell detections from a geojson file ----
    Parameters
    - args: input arguments from the argument parser
    - save_file: flag indicating if the features should be saved in an h5 file
    Outputs
    - cells_info: dataframe with the cell information. Columns ['points', 'nuclei_type', 'id', 'cell_centroid', 'cell_bbox', 'cell_feature']
    '''
    output_dir = args.output_dir
    file_anno = args.file_anno
    file_wsi = args.file_wsi
    model_dir = args.model_dir
    cell_padding = int(args.cell_padding)
    cell_masking = args.cell_masking
    
    # ---- Read the cell annotation file ----
    cell_coords, classes_color_dict = read_cell_annotation_file(file_anno)
    
    # ---- Extract cell features using Hibou-L model ----
    cells_info = extract_features_hibouL(file_wsi, cell_coords, model_dir, padding=cell_padding, masked_cell=cell_masking)
    
    if save_file:
        # ---- Save the cell and feature information in an h5 file ----
        wsi_name = os.path.splitext(os.path.basename(file_wsi))[0]
        file_cell_feature = wsi_name + '_cell_features_' + 'pad' + str(cell_padding) 
        file_cell_feature = file_cell_feature + '_masked' +'.h5' if cell_masking else file_cell_feature + '.h5'
        path_out = os.path.join(output_dir, file_cell_feature)
        save_cell_annotation_file(path_out, cells_info)
    
    return cells_info


def perform_unsupervised_clustering(cells_info, args):
    ''' ---- Perform unsupervised clustering with the methods defined and the cell embeddings ----
    Parameters
    = cells_info: dataframe with the cell information. Columns ['points', 'nuclei_type', 'id', 'cell_centroid', 'cell_bbox', 'cell_feature']
    - args: input arguments from the argument parser
    Outputs
    - mapped_int_labels: array with n labels corresponding to the unsupervised cluster mapped to the ground truth (cell nuclei_type)
    '''
    # ---- Get variables ----
    n_clusters = int(args.n_clusters)
    cluster_method = args.cluster_method
    reducer_method = args.reducer_method
    cell_features = np.vstack(cells_info['cell_feature'])
    nuclei_type = np.array(cells_info['nuclei_type'])
    
    # ---- Perform clustering -----
    labels, probabilities, score = perform_unsupervised_clustering_with_n_clusters(cell_features, n_clusters, method=cluster_method, dim_reduc=reducer_method)
    #print('[INFO] Silhoutte score: ', score)
    
    # ---- Map cluster ids to classes based on the nuclei types defined by Hibou-L ----
    mapped_int_labels, mapped_str_labels, int_nuclei_type, mapping_nuclei_type_dict = map_int_labels_to_str_labels(nuclei_type, labels)
    
    # ---- Evaluate Hibou-L predictions vs unsupervised clusters ----
    #evaluation = evaluate_gt_vs_unsupervised_clusters(int_nuclei_type, mapped_int_labels)
    
    return mapped_int_labels
    
    

if __name__ == '__main__':
    
    args = get_args()
    output_dir = args.output_dir
    file_wsi = args.file_wsi
    wsi_name = os.path.basename(file_wsi)
    n_clusters = args.n_clusters
    magnification = args.magnification
    model_name = args.model_name
    
    # ---- Check output directory exists or create it  ----
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    
    # ---- Extract cell embeddings and perform unsupervised clustering ----
    print('\n')
    print('==================================================================================')
    print('[INFO] \tUnsupervised clustering using embeddings')
    print('==================================================================================')
    print('[INFO] WSI: ', wsi_name )
    print('[INFO] Feature extraction model: ', model_name )
    print('[INFO] WSI magnification: ', magnification )
    print('[INFO] Number of clusters: ', n_clusters )
    print('[INFO] Dimensionaly reduction method: ', args.reducer_method )
    print('[INFO] Clustering method: ', args.cluster_method )
    print('[INFO] Output directory:', args.output_dir)
    print('==================================================================================')
    
    # ---- Get all cell information ----
    cells_info = extract_cell_features(args)
    
    # ---- Perform unsupervised clustering ----
    mapped_int_labels = perform_unsupervised_clustering(cells_info, args)
    
    # ---- Save the clustering info per cell in a geojson file ----
    wsi_name_no_ext = os.path.splitext(wsi_name)[0]
    path_out = os.path.join(output_dir, wsi_name_no_ext + '_'+ str(n_clusters) + 'unsupervised_clusters.geojson')
    properties_dict = {'magnification':magnification, 'cell_model':model_name, 'model_version':args.model_version, 'slide_name':wsi_name}
    save_cell_info_geoson(path_out, cells_info['id'], cells_info['points'], mapped_int_labels, properties_dict )