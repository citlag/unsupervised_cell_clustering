# Citlalli Gamez Serna

import numpy as np
import pandas as pd
import h5py
from lib.utils import read_json



def read_cell_annotation_file(anno_file):
    '''---- Read the H5 file with all cell informations ----
    Parameters:
    - anno_file  : string file path with the geojson annotation file
    Return:
    - ids: numpy array with n number of cells
    - cell_coords: dataframe n number of cells and columns with keys ['points','nuclei_type','id']
    - classes_color_dict: dictionary with class names as keys and RGB color as values. E.g. {'Neoplastic': [255, 0, 0], ...}
    '''
    
    # ---- Read the cell detections stored in the json file ----
    data = read_json(anno_file)
    
    # ---- Get and format all cell information ----
    cells = []
    classes_dict = []
    for i in range(len(data)):
        cell_type=data[i]['properties']['classification']['name']
        cell_coords = data[i]['geometry']['coordinates']
        aux = pd.DataFrame(cell_coords, columns=['points'])
        aux.insert(1,'nuclei_type', [cell_type]*len(cell_coords) )
        cells.append(aux)
        classes_dict.append(data[i]['properties']['classification'])
    
    cell_coords = pd.concat(cells)
    cell_coords = cell_coords.reset_index(drop=True)
    # Add cell id
    cell_coords['id'] = np.array(list(range(1,len(cell_coords)+1)),dtype=int)
    
    # ---- Create color dictionary of the classes ----
    classes_df = pd.DataFrame(classes_dict)
    classes_color_dict = classes_df.set_index('name')['color'].to_dict()
    # {'Neoplastic': [255, 0, 0], 'Inflammatory': [34, 221, 77], 'Connective': [35, 92, 236], 'Epithelial': [255, 159, 68]}
    
    return cell_coords, classes_color_dict



def save_cell_annotation_file(output_path, cell_info):
    '''---- Save cell information in a H5 file ----
    Parameters:
    - output_path  : string file path with h5 extension
    - cell_info: dataframe with the cell information. Columns ['points','nuclei_type','id','cell_centroid','cell_bbox','cell_feature']    
    '''
    
    # ---- Save the cell and feature information ----
    vlen_dtype = h5py.vlen_dtype(np.dtype('float32'))
    with h5py.File(output_path, "w") as f:
        features = np.stack(cell_info['cell_feature'])
        f.create_dataset("features", data=features, compression="gzip")
        # flatten each point array for storage
        coords_flat = [np.array(c, dtype=np.float32).flatten() for c in cell_info['points']]
        f.create_dataset("coords", data=coords_flat, dtype=vlen_dtype)
        f.create_dataset('centoid', data=np.stack(cell_info['cell_centroid'], dtype=float))
        f.create_dataset('bbox', data=np.stack(cell_info['cell_bbox'], dtype=float))
        f.create_dataset("ids", data=np.array(cell_info['id'], dtype=int))
        f.create_dataset('nuclei_type', data=np.array(cell_info['nuclei_type'], dtype=h5py.string_dtype(encoding='utf-8')))
        
    print('[INFO] File saved at: ', output_path)
    print('[INFO] DONE! \n')