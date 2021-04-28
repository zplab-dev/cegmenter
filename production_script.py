import pkg_resources
import pathlib
import argparse
import torch
import os
import freeimage
from elegant import datamodel

from cegmenter import production_utils

def run_predictor(exp_root, model_path=None, derived_data_path=None, pose_name='pose_cegmenter', overwrite_existing=False, img_type='png'):
    if model_path is None or not os.path.isfile(model_path):
        model_path = pkg_resources.resource_filename('cegmenter', 'models/bestValModel.paramOnly')

    exp_root = pathlib.Path(exp_root)
    if derived_data_path is None:
        derived_data_path = exp_root / 'derived_data' 

    experiment = datamodel.Experiment(exp_root)

    for position in experiment.positions.values():
        production_utils.predict_position(position, model_path, derived_data_path, pose_name, overwrite_existing,  img_type)

    if overwrite_existing:
        experiment.write_to_disk()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', action='store', type=str)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--derived_data_path', default=None)
    parser.add_argument('--pose_name', default='pose_cegmenter')
    parser.add_argument('--overwrite_existing', default=False, action='store_true')
    parser.add_argument('--img_type', default='png')

    args = parser.parse_args()
    run_predictor(**vars(args))
    
    



