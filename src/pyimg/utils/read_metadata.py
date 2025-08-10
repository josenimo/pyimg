import tifffile
import ome_types
from loguru import logger
from typing import Dict, Any

def read_metadata(file_path: str) -> Dict[str, Any]:
    """
    Reads metadata from a TIFF file, handling cases with no or partial metadata.

    Args:
        file_path: The absolute path to the image file.

    Returns:
        A dictionary containing extracted metadata:
        - 'pixel_size': Physical size of a pixel in the X dimension (float, defaults to 1.0).
        - 'channel_names': A list of channel names (list of str, defaults to []).
        - 'ome_metadata': The full OME metadata object (ome_types.OME, or None).
    """
    metadata = {
        'pixel_size': 1.0,
        'channel_names': [],
        'ome_metadata': None
    }

    try:
        logger.debug(f"Attempting to read metadata from: {file_path}")

        if not (file_path.lower().endswith('.tif') or file_path.lower().endswith('.tiff')):
            logger.warning(f"File does not have a .tif or .tiff extension: {file_path}")

        with tifffile.TiffFile(file_path) as tif:
            if tif.ome_metadata:
                ome = ome_types.from_xml(tif.ome_metadata)
                metadata['ome_metadata'] = ome
            else:
                logger.warning(f"No OME-XML metadata found in TIFF: {file_path}")
                logger.warning(f"Looking for pixel size in tags")
                page = tif.pages[0]
                if 'XResolution' in page.tags:
                    x_res_val = page.tags['XResolution'].value
                    metadata['pixel_size'] = x_res_val[1] / x_res_val[0]
                    logger.info(f"Found basic TIFF resolution tag: {metadata['pixel_size']:.4f}")
                return metadata

        # If we have an OME object, extract detailed information
        if metadata['ome_metadata']:
            ome = metadata['ome_metadata']
            image_meta = ome.images[0]

            # Safely extract pixel size
            if image_meta.pixels.physical_size_x is not None:
                metadata['pixel_size'] = image_meta.pixels.physical_size_x
                logger.info(f"Extracted pixel size: {metadata['pixel_size']:.4f} {image_meta.pixels.physical_size_x_unit}")
            else:
                logger.warning("PhysicalSizeX not found in OME metadata.")

            # Safely extract channel names
            if image_meta.pixels.channels:
                metadata['channel_names'] = [ch.name for ch in image_meta.pixels.channels if ch.name]
                if metadata['channel_names']:
                    logger.info(f"Extracted channel names: {metadata['channel_names']}")
                else:
                    logger.warning("Channels found, but they have no names in OME metadata.")
            else:
                logger.warning("No channels found in OME metadata.")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")

    return metadata