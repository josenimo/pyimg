import numpy as np
from ome_types import model
from loguru import logger
from typing import Union

def create_basic_ome(image: Union["np.ndarray", "dask.array.Array"]) -> model.OME:
    """
    Create a minimal OME metadata object from an image array.

    Args:
        image: numpy or dask array (2D, 3D, 4D, or 5D).

    Returns:
        An ome_types.model.OME object suitable for to_xml(...) / tifffile description.
    """
    shape = tuple(int(s) for s in image.shape)
    ndim = len(shape)

    if ndim == 2:
        size_c, size_z, size_t = 1, 1, 1
        size_y, size_x = shape
        dim_order = "XY"

    elif ndim == 3:
        # Heuristic: the smallest dimension is assumed to be channels (C).
        min_val = min(shape)
        min_indices = [i for i, s in enumerate(shape) if s == min_val]

        if len(min_indices) > 1:
            # Ambiguous (e.g., 10x10x10) — warn and default to CYX
            logger.warning(
                "Ambiguous 3D shape found (multiple axes share the smallest size). "
                "Defaulting to dimension order 'CYX' — please verify."
            )
            smallest_dim_idx = 0
        else:
            smallest_dim_idx = min_indices[0]

        if smallest_dim_idx == 0:
            # (C, Y, X)
            size_c, size_y, size_x = shape
            dim_order = "CYX"
        elif smallest_dim_idx == 1:
            # (Y, C, X) — uncommon
            size_y, size_c, size_x = shape
            dim_order = "YCX"
            logger.warning(f"Dimension order inferred as '{dim_order}' — verify correctness.")
        else:  # smallest_dim_idx == 2
            # (Y, X, C) — uncommon
            size_y, size_x, size_c = shape
            dim_order = "YXC"
            logger.warning(f"Dimension order inferred as '{dim_order}' — verify correctness.")

        size_z, size_t = 1, 1

    elif ndim == 4:
        # Assume Z, C, Y, X
        size_z, size_c, size_y, size_x = shape
        size_t = 1
        dim_order = "ZCYX"

    elif ndim == 5:
        # Assume T, C, Z, Y, X
        size_t, size_c, size_z, size_y, size_x = shape
        dim_order = "TCZYX"

    else:
        raise ValueError(f"Unsupported image dimensions: {shape}")

    # create channels
    channels = [
        model.Channel(id=f"Channel:{i}", samples_per_pixel=1) for i in range(int(size_c))
    ]

    pixels = model.Pixels(
        dimension_order=dim_order,
        size_x=int(size_x),
        size_y=int(size_y),
        size_c=int(size_c),
        size_z=int(size_z),
        size_t=int(size_t),
        channels=channels,
    )

    image_md = model.Image(id="Image:0", name="Scaled Image", pixels=pixels)
    ome_obj = model.OME(images=[image_md])
    return ome_obj