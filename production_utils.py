import pkg_resources
import freeimage
import numpy
import pickle
import torch

from scipy.ndimage import gaussian_filter
from skimage.measure import label

from torch.utils import data
from zplib.image import colorize
from zplib.image import pyramid
from zplib.curve import spline_geometry
from zplib.curve import interpolate

from elegant import process_images
from elegant import worm_spline
from elegant import datamodel

from elegant import convnet_spline

from cegmenter.models import WormRegMaskModel

def get_metadata(timepoint):
    metadata = timepoint.position.experiment.metadata
    try:
        objective, optocoupler, temp = metadata['objective'], metadata['optocoupler'], metadata['nominal_temperature']
    except KeyError:
        objective = 5
        optocoupler = 1
        temp = 25
    magnification = objective * optocoupler
    return objective, optocoupler, magnification, temp

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == numpy.argmax(numpy.bincount(labels.flat)[1:])+1
    return largestCC

def get_output_images(out):
    mask_CNN = getLargestCC(out[('Mask',0)].cpu().detach().numpy()[0][0] > 0.5)
    xcoord_CNN = out[('X_Coord',0)].cpu().detach().numpy()[0][0]*mask_CNN
    ycoord_CNN = out[('Y_Coord',0)].cpu().detach().numpy()[0][0]*mask_CNN
    return xcoord_CNN, ycoord_CNN, mask_CNN

def preprocess_image(timepoint):
    img_path = timepoint.image_path('bf')
    lab_frame_image = freeimage.read(img_path)
    lab_frame_image = lab_frame_image.astype(numpy.float32)
    height, width = lab_frame_image.shape[:2]
    objective, optocoupler, magnification, temp = get_metadata(timepoint)

    #mode = process_images.get_image_mode(lab_frame_image, optocoupler=optocoupler)
    mode = process_images.get_image_mode(lab_frame_image)
    return (lab_frame_image - mode)/mode

def pad_image(image):
    image_shape = image.shape
    if image_shape[0]%32 != 0 or image_shape[1]%32 !=0:
        offsetx = image_shape[0]%32
        offsety = image_shape[1]%32
        pxo = int(offsetx/2)
        pyo = int(offsety/2)

        padded_image = numpy.pad(image, ((pxo, pxo),(pyo,pyo)), 'edge')
    else:
        padded_image = image
        pxo = 0
        pyo = 0
    
    return padded_image, pxo, pyo

def crop_image(padded_image, pxo, pyo):
    height, width = padded_image.shape
    return padded_image[pxo:height-pxo, pyo:width-pyo]

def predict_timepoint(timepoint, model_path):
    lab_frame_image = preprocess_image(timepoint)
    #ensure image is the correct dimensions for the CNN
    padded_image, pxo, pyo = pad_image(lab_frame_image)

    extend_image = numpy.stack([padded_image, padded_image, padded_image])
    outputs = predict_image(extend_image, model_path)
    ap_coords, dv_coords, mask = outputs 
    #crop the outputs to be the original lab_frame_image shape
    ap_coords = crop_image(ap_coords, pxo, pyo)
    dv_coords = crop_image(dv_coords, pxo, pyo)
    mask = crop_image(mask, pxo, pyo)  

    costs, centerline, center_path, pose = convnet_spline.find_centerline(ap_coords, dv_coords, mask)
    timepoint.annotations['pose_cegmenter'] = pose
    return pose, ap_coords, dv_coords, mask

def predict_image(image, model_path):
    regModel = WormRegMaskModel.WormRegModel(34, pretrained=True)
    regModel.load_state_dict(torch.load(model_path, map_location='cpu'))
    regModel.eval()

    tensor_img = torch.tensor(image).unsqueeze(0)
    out = regModel(tensor_img)
    ap_coords, dv_coords, mask = get_output_images(out)
    return ap_coords, dv_coords, mask
