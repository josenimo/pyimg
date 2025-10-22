from dask_image import imread
import tifffile
import os
from loguru import logger
import argparse
from pathlib import Path
import numpy as np
import ome_types

from pyimg.utils import batch_process, describe_image, read_metadata


def project (image_path, output_path, binning_factor:int = 2):

    # describe input
    describe_image(image_path)

    # load and change metadata
    ome_metadata = read_metadata(image_path)
    # unsure how to change binning
    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')

    image = imread.imread(image_path)
    image = image.rechunk(chunks={0:1, 1:512, 2:512})



### INPUT parsing
# default use argparse
# Use a csv file with marker list, and column selection nucleus and membrane





def scale_2D_numpy_array(array, min_max_quantiles=[0.0, 1.0]):
    """Scale 2d numpy array to 8-bit with user chosen quantiles"""
    q_min = np.quantile(array, min_max_quantiles[0])
    q_max = np.quantile(array, min_max_quantiles[1])
    clipped_array = np.clip(array, q_min, q_max)
    rescaled_array = (clipped_array - q_min) / (q_max - q_min) * 255.0
    rescaled_array_8bit = rescaled_array.astype(np.uint8)
    return rescaled_array_8bit

def scale_image_channels(image, min_max_quantiles=[0.0, 1.0]):
    """Apply the scaling function to each channel of a (C, Y, X) image"""
    assert image.shape[0] == np.min(image.shape), "Image must with shape (C, Y, X)"
    scaled_channels = []
    for c in range(image.shape[0]):
        scaled_channel = scale_2D_numpy_array(image[c], min_max_quantiles)
        scaled_channels.append(scaled_channel)
    return np.stack(scaled_channels, axis=0)

def projection(image, channels, quantile=0.9, min_max_quantiles=[0.0, 1.0]):
    """Project the channels from a C,Y,X image"""
    logger.info(f"Creating projection for channels: {channels}")
    selected_channels_image = image[channels]
    scaled_image = scale_image_channels(selected_channels_image, min_max_quantiles)
    quantile_projection = np.quantile(scaled_image, quantile, axis=0).astype(np.uint8)
    return quantile_projection



def main():
    parser = argparse.ArgumentParser(description="Scale the data type of a 3D image or a directory of images.")
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Input image file or directory path.")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file or directory path.")
    parser.add_argument("-f", "--force", dest="force", action="store", help="Force overwrite of existing files.")

    args = parser.parse_args()

    check_inputs_paths(args)

    func_kwargs = {'binning_factor': args.binning_factor}

    batch_process(
        # process_func=binning,
        input_path=args.input,
        output_path=args.output,
        func_kwargs=func_kwargs,
        force=args.force,
        output_suffix="_bin"
    )

if __name__ == "__main__":
    main()



def get_args():
    """Get arguments from command line"""
    description = """Create a projection of the nucleus and membrane channels for cellpose segmentation"""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--input",        dest="input",       action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output",       dest="output",      action="store", required=True, help="Path to output image.")
    inputs.add_argument("-cn", "--channels_nucleus",     dest="channels_nucleus",    action="store", required=True, type=int, nargs='+', help="Channels to project for nucleus(0-based)")
    inputs.add_argument("-cm", "--channels_membrane",     dest="channels_membrane",    action="store", required=True, type=int, nargs='+', help="Channels to project for membrane(0-based)")
    inputs.add_argument("-nq", "--nuclear_quantile",    dest="nuclear_quantile",    action="store", required=True, type=float, help="Quantile value for nuclear projection")
    inputs.add_argument("-mq", "--membrane_quantile",   dest="membrane_quantile",    action="store", required=True, type=float, help="Quantile value for membrane projection")
    inputs.add_argument("-mmq", "--membrane_min_max_scaling_quantiles", dest="membrane_min_max_scaling_quantiles", action="store", required=False, type=str, nargs='+', help="Min and max quantiles for membrane projection")
    inputs.add_argument("-nmq", "--nucleus_min_max_scaling_quantiles",     dest="nucleus_min_max_scaling_quantiles",    action="store", required=False, type=str, nargs='+', help="Min and max quantiles for membrane projection")    
    inputs.add_argument("-ll", "--log-level",   dest="loglevel",    default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')

    arg = parser.parse_args()
    arg.input = os.path.abspath(arg.input)
    arg.output = os.path.abspath(arg.output)
    if arg.membrane_min_max_scaling_quantiles:
        arg.membrane_min_max_scaling_quantiles = [float(x) for x in arg.membrane_min_max_scaling_quantiles]
    if arg.nucleus_min_max_scaling_quantiles:
        arg.nucleus_min_max_scaling_quantiles = [float(x) for x in arg.nucleus_min_max_scaling_quantiles]
    return arg


def check_inputs_paths(args):
    """ check inputs and outputs """
    #input
    assert os.path.isfile(args.input), "Input file does not exist"
    assert args.input.endswith((".tif",".tiff")), "Input file must be a .tif or .tiff file"
    #output
    assert args.output.endswith((".tif",".tiff")), "Output file must be a .tif file"
    #channels_nucleus
    assert len(args.channels_nucleus) > 0, "No channels provided for nucleus"
    assert all([x >= 0 for x in args.channels_nucleus]), "All channel numbers must be positive"
    #channels_membrane
    assert len(args.channels_membrane) > 0, "No channels provided for membrane"
    assert all([x >= 0 for x in args.channels_membrane]), "All channel numbers must be positive"
    #nuclear quantile
    assert 0 <= args.nuclear_quantile <= 1, "Nuclear quantile must be between 0 and 1"
    #membrane quantile
    assert 0 <= args.membrane_quantile <= 1, "Membrane quantile must be between 0 and 1"
    #membrane min max quantiles
    if args.membrane_min_max_scaling_quantiles:
        assert len(args.membrane_min_max_scaling_quantiles) == 2, "Membrane min max quantiles must be a list of 2 values"
        assert all([0 <= float(x) <= 1 for x in args.membrane_min_max_scaling_quantiles]), "Membrane min max quantiles must be between 0 and 1"
    #nuclear min max quantiles
    if args.nucleus_min_max_scaling_quantiles:
        assert len(args.nucleus_min_max_scaling_quantiles) == 2, "Nuclear min max quantiles must be a list of 2 values"
        assert all([0 <= float(x) <= 1 for x in args.nucleus_min_max_scaling_quantiles]), "Nuclear min max quantiles must be between 0 and 1"
    #log level
    assert args.loglevel in ["DEBUG", "INFO"], "Log level must be either DEBUG or INFO"