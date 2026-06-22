import numpy as np
import tifffile
import ome_types
from pathlib import Path
from typing import List, Optional, Tuple, Union
from ome_types.model import PixelType

def create_test_image(
    path: Union[str, Path],
    shape: Tuple[int, ...],
    dtype: Union[str, np.dtype] = np.uint16,
    layout: str = "CYX",
    has_metadata: bool = True,
    channel_names: Optional[List[str]] = None,
    pixel_size: Optional[float] = None,
    pixel_size_unit: str = "µm"
) -> Path:
    """
    Generates a test TIFF or OME-TIFF image with specified properties.

    Args:
        path: Path where the image file will be saved.
        shape: Shape of the image array. Must match the layout length.
        dtype: Data type of the image.
        layout: Order of dimensions (e.g. 'XY', 'CYX', 'CXY', 'TCZYX').
        has_metadata: If True, writes OME-XML description metadata.
        channel_names: Optional list of names for each channel.
        pixel_size: Physical size of pixels (applied to X and Y).
        pixel_size_unit: Unit for physical pixel size (default 'µm').

    Returns:
        The Path to the created file.
    """
    path = Path(path)
    dtype = np.dtype(dtype)

    # 1. Generate array data
    if np.issubdtype(dtype, np.integer):
        try:
            info = np.iinfo(dtype)
            max_val = min(1000, info.max)
        except ValueError:
            max_val = 100
        image_data = np.random.randint(0, max_val, size=shape, dtype=dtype)
    else:
        image_data = np.random.random(size=shape).astype(dtype)

    # 2. Handle OME-XML metadata if required
    if has_metadata:
        layout_upper = layout.upper()
        # Parse sizes from layout and shape
        sizes = {'X': 1, 'Y': 1, 'C': 1, 'Z': 1, 'T': 1}
        for char, size in zip(layout_upper, shape):
            if char in sizes:
                sizes[char] = size

        # Map numpy dtype string to OME PixelType
        dtype_str = dtype.name
        numpy_to_ome = {
            "int8": PixelType.INT8,
            "int16": PixelType.INT16,
            "int32": PixelType.INT32,
            "uint8": PixelType.UINT8,
            "uint16": PixelType.UINT16,
            "uint32": PixelType.UINT32,
            "float32": PixelType.FLOAT,
            "float64": PixelType.DOUBLE,
        }
        ome_pixel_type = numpy_to_ome.get(dtype_str, PixelType.UINT16)

        ome = ome_types.model.OME()
        
        # Build Channels
        channels = []
        if channel_names:
            for idx, name in enumerate(channel_names):
                channels.append(ome_types.model.Channel(id=f"Channel:0:{idx}", name=name))
        else:
            for idx in range(sizes['C']):
                channels.append(ome_types.model.Channel(id=f"Channel:0:{idx}", name=f"Channel_{idx}"))

        # Build Pixels
        pixels = ome_types.model.Pixels(
            id="Pixels:0",
            dimension_order="XYCZT",  # Standard OME default dimension order
            type=ome_pixel_type,
            size_x=sizes['X'],
            size_y=sizes['Y'],
            size_c=sizes['C'],
            size_z=sizes['Z'],
            size_t=sizes['T'],
            channels=channels
        )

        if pixel_size is not None:
            pixels.physical_size_x = pixel_size
            pixels.physical_size_y = pixel_size
            if pixel_size_unit:
                pixels.physical_size_x_unit = pixel_size_unit
                pixels.physical_size_y_unit = pixel_size_unit

        ome.images.append(
            ome_types.model.Image(
                id="Image:0",
                pixels=pixels
            )
        )

        ome_xml = ome_types.to_xml(ome)
        tifffile.imwrite(path, image_data, description=ome_xml.encode('utf-8'))
    else:
        tifffile.imwrite(path, image_data)

    return path
