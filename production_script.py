import pkg_resources
import pathlib
import argparse
import torch
import os
import freeimage
from elegant import datamodel

from cegmenter import production_utils

def run_predictor(exp_root, model_path=None, derived_data_path=None, pose_name='pose_cegmenter', overwrite_existing=False):
	if model_path is None:
		with pkg_resources.resource_stream('cegmenter', 'models/bestValModel.paramOnly') as m:
			model_path = m

	if not os.path.isfile(model_path):
		with pkg_resources.resource_stream('cegmenter', 'models/bestValModel.paramOnly') as m:
			model_path = m

	exp_root = pathlib.Path(exp_root)
	if derived_data_path is None:
		derived_data_path = exp_root / 'derived_data' 

	experiment = datamodel.Experiment(exp_root)



