# Citlalli Gamez Serna


import os, json, h5py
import numpy as np


def read_json(file):
    '''---- Read json file ----
    Parameters:
    - file  : string file path
    Return:
    - data
    '''
    with open(file) as f:
        data = json.load(f)
    return data


def read_cell_h5_file(file):
    '''---- Read the H5 file with all cell informations ----
    Parameters:
    - file  : string file path
    Return:
    - ids: numpy array with n number of cells
    - cell_coords: numpy array of cell coordinates (n,2)
    - cell_features: numpy array with cell embedings (n,1024)
    - nuclei_type: numpy array with the string names of each cell type 
    - bboxs: numpy array with n cell bboxes. Each bounding box consist of the top left and right bottom corner [x1,y1,x2,y2]
    - centroids: numpy array with n cell centroids [x,y]
    '''
    with h5py.File(file, "r") as f:
        coords_vlen = f["coords"][:]
        nuclei_type = f['nuclei_type'][:]
        features = f['features'][:]
        ids = f['ids'][:]
        #centroids = f['centroid'][:]
        bboxs = f['bbox'][:]
    
    nuclei_type = nuclei_type.astype(str)
    # reshape cell coordinates to be in a array of (n,2)
    cell_coords = [c.reshape(-1, 2) for c in coords_vlen]
    cell_features = np.array(features)
    #return ids, cell_coords, cell_features, nuclei_type, bboxs, centroids
    return ids, cell_coords, cell_features, nuclei_type, bboxs