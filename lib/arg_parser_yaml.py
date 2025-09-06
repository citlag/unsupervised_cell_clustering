# Citlalli Gamez Serna

import os
import sys
import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(
                                        prog='extract_features_cluster',
                                        description='Extract cell features and perform unsupervised clustering',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                        ,add_help=False
                                    )
    
    parser.add_argument('-config', '--config_file', dest='config_file', required=True, help='Specify .yaml configuration file')
    
    parser.add_argument('-output', '--out_dir', dest='output_dir', required=True, help='Specify directory to save the outputs')
    
    parser.add_argument('-wsi', '--wsi_path', dest='file_wsi', required=True, help='Path to wsi reference extract the cell features.')
    
    parser.add_argument('-anno', '--anno_path', dest='file_anno', required=True, help='Path to the geojson annotation file containing the cell detections.')

    parser.add_argument('-model', '--model_dir', dest='model_dir', required=True, help='Directory to deep learning model to extract the cell features/embeddings.')
    
    args = parser.parse_args()
    
    # Read the configuration yaml file
    if os.path.isfile( args.config_file ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input yaml configuration file ' + args.config_file + ' does not exist!\n')
    else:    
        with open(args.config_file) as config_file:
            yaml_config = yaml.load(config_file, Loader=yaml.FullLoader)
            print('[INFO] Arguments taken from: {}'.format(args.config_file) )
        for dict_key in yaml_config:
            print('\t{} : {} '.format(dict_key, yaml_config[dict_key]))
            setattr(args, dict_key, yaml_config[dict_key])
    
    
    if args.file_wsi is None:
        parser.print_help()
        sys.exit('\nERROR: Input WSI file not specified!\n')
        
    if os.path.isfile( args.file_wsi ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input WSI file ' + args.file_wsi + ' does not exist!\n')
    
    if args.file_anno is None:
        parser.print_help()
        sys.exit('\nERROR: Input cell annotation file not specified!\n')
    
    if os.path.isfile( args.file_anno ) is False:
        parser.print_help()
        sys.exit('\nERROR: Input cell annotation file ' + args.file_anno + ' does not exist!\n')
    
    if args.output_dir is None:
        parser.print_help()
        sys.exit('\nERROR: Output directory not specified!\n')
    
    if args.model_dir is None:
        parser.print_help()
        sys.exit('\nERROR: Input model directory has not been specified!\n')
    
    if args.cell_padding is None or not isinstance(int(args.cell_padding), int):
        parser.print_help()
        sys.exit('\nERROR: Cell padding has not been specified or it is not an integer value!\n')
    
    if args.cell_masking is None or not isinstance(args.cell_masking, bool):
        parser.print_help()
        sys.exit('\nERROR: Cell masking has not been specified or is not True or False!\n')
    
    if args.magnification is None:
        parser.print_help()
        sys.exit('\nERROR: Magnification has not been specified!\n')
    
    if args.model_name is None:
        parser.print_help()
        sys.exit('\nERROR: Model name has not been specified!\n')
    
    if args.model_version is None:
        parser.print_help()
        sys.exit('\nERROR: Model version has not been specified!\n')
    
    if args.n_clusters is None or int(args.n_clusters) < 2:
        parser.print_help()
        sys.exit('\nERROR: Number of clusters has not been specified or is less than 2!\n')
    
    if args.reducer_method is None:
        parser.print_help()
        sys.exit('\nERROR: Reducer has not been specified!\n')
    
    reducer_arg_names = ['PCA', 'UMAP']   
    if not args.reducer_method in reducer_arg_names:
        sys.exit('\nERROR: Reducer name not allowed!\n')
    
    if args.cluster_method is None:
        parser.print_help()
        sys.exit('\nERROR: Cluster method has not been specified!\n')
    
    cluster_arg_names = ['KMeans', 'GMM', 'Spectral']   
    if not args.cluster_method in cluster_arg_names:
        sys.exit('\nERROR: Cluster method name not allowed!\n')
    
    return args
