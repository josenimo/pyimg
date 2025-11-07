import dask.array as da
import numpy as np
from typing import List, Dict, Any, Union
import argparse
import pandas as pd
from dask_image import imread
import tifffile
from loguru import logger
import ome_types

from pyimg.utils import batch_process, describe_image, read_metadata

#TODO sum is a bad default, high quantile, like 80%, therefore it will usually pick max value until 10 markers 
#

def config_from_csv(markers_csv_path: str, projection_method: str = "sum") -> List[Dict[str, Any]]:
    """
    Parses a markers CSV file using pandas to generate the projection_config.
    """
    try:
        df = pd.read_csv(markers_csv_path)
    except FileNotFoundError:
        logger.error(f"Markers CSV file not found at: {markers_csv_path}")
        raise

    df = df[df['projection'].notna() & (df['projection'].str.strip() != '')]
    projection_groups = df.groupby('projection')['Channel_number'].apply(list)

    # list of dictionaries
    projection_config = []
    output_channel_counter = 0
    for group_name, channels in projection_groups.sort_index().items():
        config_item = {
            "output_channel": output_channel_counter,
            "input_channels": sorted([c - 1 for c in channels]),
            "projection": projection_method,
            "kwargs": {}
        }
        projection_config.append(config_item)
        output_channel_counter += 1
        logger.info(f"Found group '{group_name}' with channels {channels}")

    return projection_config

def _validate_projection_config(
    projection_config: List[Dict[str, Any]],
    num_in_channels: int
) -> int:
    """Helper to validate the projection configuration."""
    if not projection_config:
        raise ValueError("Projection config list cannot be empty.")

    num_out_channels = len(projection_config)
    if num_out_channels > 3:
        print(f"Warning: Output has {num_out_channels} channels, which is > 3.")

    output_channel_indices = {p.get("output_channel") for p in projection_config}
    if None in output_channel_indices:
        raise ValueError("Each projection dictionary must have an 'output_channel' key.")
    if len(output_channel_indices) != num_out_channels:
        raise ValueError("Output channel indices must be unique.")
    if set(range(num_out_channels)) != output_channel_indices:
        raise ValueError(
            f"Output channel indices must be a contiguous range from 0 to N-1. "
            f"Got: {sorted(list(output_channel_indices))}"
        )

    used_in_indices: set[int] = set()
    for proj in projection_config:
        in_indices = proj.get("input_channels")
        if not in_indices:
            raise ValueError(f"Input channel list for output channel {proj['output_channel']} is empty.")

        for idx in in_indices:
            if not (0 <= idx < num_in_channels):
                raise IndexError(
                    f"Input channel index {idx} (for output channel {proj['output_channel']}) "
                    f"is out of bounds. Input image has {num_in_channels} channels (0 to {num_in_channels-1})."
                )
            if idx in used_in_indices:
                raise ValueError(
                    f"Input channel {idx} is used more than once (violates exclusivity)."
                )
            used_in_indices.add(idx)

    return num_out_channels

def scale_channels_dask(
    img_stack: da.Array,
    min_max_quantiles: Union[List[float], tuple[float, float]] = (0.01, 0.999)
) -> da.Array:
    """
    Lazily scale each channel of a Dask array based on quantiles.
    The output is scaled to the range of the input dtype.
    """
    if img_stack.ndim != 3:
        raise ValueError(f"Input array must be 3D (C, Y, X). Got {img_stack.ndim} dimensions.")

    def scale_single_channel(channel):
        q_min, q_max = da.percentile(channel, [min_max_quantiles[0] * 100, min_max_quantiles[1] * 100])
        clipped = da.clip(channel, q_min, q_max)
        scaled = (clipped - q_min) / (q_max - q_min)
        return scaled

    scaled_channels = [scale_single_channel(img_stack[i]) for i in range(img_stack.shape[0])]
    
    output_stack = da.stack(scaled_channels, axis=0)
    return output_stack


def merge_channels_dask(
    img_stack: da.Array,
    projection_config: List[Dict[str, Any]],
    scale_before_project: bool = True,
    scale_quantiles: Union[List[float], tuple[float, float]] = (0.01, 0.999)
) -> da.Array:
    """
    Lazily merge subsets of channels from a (C, Y, X) Dask array based on a flexible configuration.
    """
    if img_stack.ndim != 3:
        raise ValueError(f"Input array must be 3D (C, Y, X). Got {img_stack.ndim} dimensions.")

    num_in_channels = img_stack.shape[0]
    _validate_projection_config(projection_config, num_in_channels)

    output_channels_list = []

    for proj_info in sorted(projection_config, key=lambda p: p["output_channel"]):
        in_indices = proj_info["input_channels"]
        proj_method = proj_info.get("projection", "sum")
        proj_kwargs = proj_info.get("kwargs", {})

        selected_channels = img_stack[in_indices, :, :]

        if scale_before_project:
            selected_channels = scale_channels_dask(selected_channels, min_max_quantiles=scale_quantiles)

        if proj_method == "sum":
            merged_channel = selected_channels.sum(axis=0, **proj_kwargs)
        elif proj_method == "mean":
            merged_channel = selected_channels.mean(axis=0, **proj_kwargs)
        elif proj_method == "max":
            merged_channel = selected_channels.max(axis=0, **proj_kwargs)
        else:
            raise ValueError(f"Unknown projection method: {proj_method}. Use 'sum', 'mean', or 'max'.")

        output_channels_list.append(merged_channel)

    output_stack = da.stack(output_channels_list, axis=0)
    return output_stack

def main():
    parser = argparse.ArgumentParser(description="Project channels of a TIFF image using a CSV markers file.")
    parser.add_argument("-i", "--input", required=True, help="Input image file or directory.")
    parser.add_argument("-o", "--output", required=True, help="Output file or directory.")
    parser.add_argument("--markers-csv", required=True, help="Path to the markers CSV file.")
    parser.add_argument(
        "--projection-method",
        type=str,
        default="sum",
        choices=["sum", "mean", "max"],
        help="The projection method to apply to all groups (default: sum)."
    )
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite of existing files.")

    args = parser.parse_args()

    def process_image(image_path, output_path):
        describe_image(image_path)
        
        try:
            projection_config = config_from_csv(args.markers_csv, args.projection_method)
        except FileNotFoundError:
            return

        if not projection_config:
            logger.warning("Projection config is empty. No output will be generated.")
            return

        img = imread.imread(image_path)
        
        projected_img = merge_channels_dask(
            img,
            projection_config=projection_config,
            scale_before_project=True,
            scale_quantiles=(0.01, 0.999)
        )
        
        try:
            ome_metadata = read_metadata(image_path)
            if ome_metadata:
                ome_metadata.images[0].pixels.size_c = len(projection_config)
                ome_metadata.images[0].pixels.channels = ome_metadata.images[0].pixels.channels[:len(projection_config)]
                ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')
            else:
                ome_xml_bytes = None
        except Exception as e:
            logger.warning(f"Could not update OME metadata: {e}")
            ome_xml_bytes = None

        logger.info(f"Writing projected image to '{output_path}'...")
        tifffile.imwrite(output_path, projected_img.compute(), bigtiff=True, description=ome_xml_bytes)
        logger.success(f"Successfully created '{output_path}' ✨")

    batch_process(
        process_func=lambda inp, outp: process_image(inp, outp),
        input_path=args.input,
        output_path=args.output,
        func_kwargs={},
        force=args.force,
        output_suffix="_projected"
    )

if __name__ == "__main__":
    main()