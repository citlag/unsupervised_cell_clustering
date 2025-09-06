# Citlalli Gamez Serna

import torch
import pandas as pd
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import cv2
from PIL import Image
import large_image



def to_rgb(img):
    # ---- Convert image to RGB ----
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img



def bbox_from_coords(coords, max_size, pad=12):
    ''' ---- Get the cell centroid and bounding box ----
    Parameters
    - coords: list or array of xy coordinates. E.g. [[1,2],[3,4]]
    - pad: padding to consider for the bounding box. Default is 12 pixels
    - max_size: tuple of x,y with the maximun size to consider
    Outputs
    - bounding box coordinates [x1,y1,x2,y2]
    - centroid [x,y]
    '''
    coords_aux = coords.copy()
    if len(coords_aux) == 0:
        return None
    if isinstance(coords_aux, list):
        coords_aux = np.array(coords_aux)
    if coords_aux.shape[1]<2:        
        print('[INFO] Coordinates in the wrong format!')
        return None
    # Replace values less than 0 to zero
    invalid_coords_index = coords_aux<0
    coords_aux[invalid_coords_index] = 0
    # Get the top-left and bottom-right points
    x1 = min(coords_aux[:,0]); y1 = min(coords_aux[:,1])
    x2 = max(coords_aux[:,0]); y2 = max(coords_aux[:,1])
    centroid = [(x2-x1)/2 + x1, (y2-y1)/2 + y1]
    if pad > 0:
        x1 = max(0, x1 - pad ); y1 = max(0, y1 - pad)
        x2 = min(max_size[0], x2 + pad); y2 = min(max_size[1], y2 + pad)
    bbox = [x1,y1,x2,y2]    
    return bbox, centroid


def extract_binary_cell_mask(cell_coords, bbox):
    ''' ---- Extract the binary cell mask ----
    Parameters
    - cell_coords: list or array of xy coordinates. E.g. [[1,2],[3,4]]
    - bbox: list of 4 elements indicating [x1,y1,x2,y2]
    Outputs
    - mask: binary mask of the cell
    '''
    #mask_size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
    mask_size = ( bbox[3]-bbox[1], bbox[2]-bbox[0] )
    mask = np.zeros(mask_size, dtype=np.uint8)
    cell_coords_ref = cell_coords.copy()
    if isinstance(cell_coords_ref, list):
        cell_coords_ref = np.array(cell_coords_ref)
    cell_coords_ref[:,0] = cell_coords_ref[:,0]-bbox[0]
    cell_coords_ref[:,1] = cell_coords_ref[:,1]-bbox[1]
    cv2.fillPoly(mask, [cell_coords_ref], color=1)
    return mask
    

def extract_cell_image(cell_coords, slide, max_image_size, pad=12, masked=False):
    ''' ---- Extract the RGB image of the cell using the cell coordinates ----
    Parameters
    - cell_coords: list or array of xy coordinates. E.g. [[1,2],[3,4]]
    - max_image_size: tuple of x,y with the maximun size to consider
    - pad: padding in pixels to consider around the cell
    - masked: bool flag indicating if the image should be masked to zero values to remove surrounding tissue of the cell
    Outputs
    - im_cell: rgb image cell
    = bbox
    = centroid 
    '''
    bbox,centroid = bbox_from_coords(cell_coords,max_image_size,pad=pad)
    # - Extract the cell_image -
    im_cell, _ = slide.getRegion( region=dict(left=bbox[0], top=bbox[1], width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], units='base_pixels'),
    format=large_image.tilesource.TILE_FORMAT_NUMPY, )
    if masked:
        cell_mask = extract_binary_cell_mask(cell_coords,bbox)
        #im_cell_masked = im_cell.copy()
        im_cell[cell_mask == 0] = 0  # zero out background
    return to_rgb(im_cell), bbox, centroid



def get_cell_feature(cell_crop, model, processor, device):
    ''' ---- Extract the cell feature using Hibou-L ----
    Parameters
    - cell_crop: NumPy image (H, W, C), e.g., RGB uint8
    - model: torch Hibou-L model loaded
    - processor: torch processor
    - device: device to use for the model prediction
    Outputs
    - feat: array of the image feature with size 1024
    '''
    # Convert to PIL Image if needed
    img = Image.fromarray(cell_crop)
    
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Inputs include pixel_values already normalized
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Usually: outputs.last_hidden_state or pooled_output depending on model
        feat = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    
    return feat.squeeze().cpu().numpy()





def extract_features_hibouL(file_wsi, cell_info, model_dir, padding=0, masked_cell=False):
    ''' ---- Extract the cell features ----
    Parameters
    file_wsi: str path of the WSI
    cell_info: dataframe with n cell information. Columns ['points','nuclei_type','ids']
    model_dir: str directory path to the Hibou-L model
    Outputs
    cells_info: copy of the cell_info dataframe with the additional columns ['cell_centroid','cell_bbox','cell_feature']
    '''
    # ---- Read the WSI ----
    slide = large_image.getTileSource( file_wsi )
    metadata = slide.getMetadata()
    slide_base_size = [metadata['sizeX'], metadata['sizeY']]
    
    # ---- Load Hibou-L ---
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ---- Get the cell image and embedding for feature extraction ----
    bboxes = []; centroids = []; features=[]
    for i in range(len(cell_info)):
        xy = cell_info.iloc[i]['points']
        im_cell, bbox, centroid = extract_cell_image(xy, slide, slide_base_size, pad=padding, masked=masked_cell)
        cell_feat = get_cell_feature(im_cell, model, processor, device)
        bboxes.append(bbox); centroids.append(centroid); features.append(cell_feat)
    
    # ---- Concatenate the outputs ----
    df_aux = pd.DataFrame(list(zip(centroids,bboxes,features)), columns=['cell_centroid','cell_bbox','cell_feature'])
    cells_info = pd.concat([cell_info.reset_index(drop=True),df_aux.reset_index(drop=True)], axis=1)
    
    return cells_info