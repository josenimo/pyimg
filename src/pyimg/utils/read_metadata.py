import ome_types
from loguru import logger
from typing import Optional

def read_metadata(file_path: str) -> Optional[ome_types.OME]:
    """
    Reads metadata from a TIFF file, handling cases with no or partial metadata.
    Args:
        file_path: The absolute path to the image file.
    Returns:
        ome_types.OME object
    """   
    try:
        ome = ome_types.from_tiff(path = file_path)
        return ome
    except ValueError as e:
        logger.warning(f"Could not parse OME-XML metadata. "
                        f"The file may not contain it or it's malformed.")
        logger.debug(f"Underlying error: {e}")
        return None
        