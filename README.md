# Cell Unsupervised Clustering

This is an agile repository to perform cell unsupervised clustering using Hibou-L model to extract the cell embeddings.  
Each embedding has a size of 1024 for which dimensionality reductions is also required.

The unsupervised clustering exploration allows to chose between 2 dimensionality reduction methods (PCA and UMAP) and 3 clustering methods (KMeans, Gaussian Mixture Models and Spectral clustering) from sklearn library.
The implementation requires the user to define the number of clusters.

## Main files
* **read_cell_detections_perform_clustering.py** Script to execute the cell feature extraction and unsupervised clustering 
* **config.yaml** Configuration yaml file to be used as input for the python script. 

### Configuration file ###
The **config.yaml** is used as input for the python script and should contain the following arguments:
* *model_dir* -> string. Directory where Hibou-L model is stored.
* *cell_padding* -> int. Number of pixels to pad the cell image.
* *cell_masking*  -> bool. Indicate the cell image should be masked for the feature extraction.
* *magnification*  -> float. Magnification to use to extract the cell images.
* *model_name*  -> string. Model name used to extract the cell embeddings.
* *model_version*  -> string. Version of the code.
* *n_clusters*  -> int. Number of clusters (>=2). 
* *cluster_method*  -> sting. Name of the unsupervised clustering method. Available options are KMeans (KMeans), Spectral clustering (Spectral) and Gaussian Mixture Model (GMM) from sklearn.
* *reducer_method*  -> string. Name of the dimensionality reduction method. Available methods are PCA and UMAP.

*NOTE:* in the magnification argument, you should use the base magnification of the image as of today version the code does not perform coordinate conversion from magnification x to magnification y.
<br>


## Running example

```
python read_cell_detections_perform_clustering.py \
	-config /path/to/configuration/yamlFileConfiguration 
	-output /path/to/outputDirectory
	-wsi /path/to/wsi/WSI.svs
	-anno /path/to/cellDetections/WSI_annotations.geojson
```

The script will save in the output_path:
* a H5 file which contains all cell information ['points', 'nuclei_type', 'id', 'cell_centroid', 'cell_bbox', 'cell_feature'];
* a GEOJSON file with the WSI cell information including the unsupervised cluster per cell.

GEOJSON structure:
```
{
	"type": "Feature",
	"geometry": {
		"type": "Polygon",
		"coordinates": [ ... ] // WSI coordinates of the cell's boundary
	},
	"properties": {
		"id": "unique_cell_id",
		"model_label": "Cluster X", // Where X is your unsupervised cluster ID (1-n)
		"model_class_id": X, // The integer ID of the cluster
		"model_magnification": 40, // The magnification at which you processed the slide
		"class_type": "object",
		"model_name": "HibouLCellVIT",
		"model_version": "1.0",
		"anet_class_label": "Cluster X", // Same as model_label
		"slide": "slide_name.svs"
	}
}
```

<br>

## Author
* **Citlalli G&aacute;mez Serna** - September 2025

<br>

### Last Update
03.09.2025