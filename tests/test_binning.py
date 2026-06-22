import pytest
import numpy as np
import tifffile
import ome_types
from pathlib import Path

from pyimg.binning import binning
from pyimg.utils.read_metadata import read_metadata
from .helpers import create_test_image

@pytest.mark.parametrize(
    "shape,layout,dtype,has_metadata,channel_names,pixel_size,binning_factor",
    [
        # XY (2D), standard uint16, with metadata, binning factor 2
        ((50, 60), "XY", np.uint16, True, ["Gray"], 0.8, 2),
        # XY (2D), standard uint16, without metadata, binning factor 2
        ((50, 60), "XY", np.uint16, False, None, None, 2),
        # CYX (3D), standard uint16, with metadata, binning factor 2
        ((3, 50, 60), "CYX", np.uint16, True, ["DAPI", "FITC", "Cy5"], 1.2, 2),
        # CYX (3D), standard uint16, without metadata, binning factor 2
        ((3, 50, 60), "CYX", np.uint16, False, None, None, 2),
        # CXY (3D), standard uint16, with metadata, binning factor 3
        ((3, 60, 90), "CXY", np.uint16, True, ["Ch1", "Ch2", "Ch3"], 1.5, 3),
        # CYX (3D), float32, with metadata, binning factor 2
        ((2, 40, 40), "CYX", np.float32, True, ["Red", "Green"], 0.5, 2),
    ]
)
def test_image_binning(
    tmp_path, shape, layout, dtype, has_metadata, channel_names, pixel_size, binning_factor
):
    """Test binning images across layouts, metadata options, and factors."""
    input_path = tmp_path / f"test_img_{layout}_{binning_factor}.tif"
    output_path = tmp_path / "binned_output.tif"

    # Create test image
    create_test_image(
        path=input_path,
        shape=shape,
        dtype=dtype,
        layout=layout,
        has_metadata=has_metadata,
        channel_names=channel_names,
        pixel_size=pixel_size
    )

    # Run binning
    binning(str(input_path), str(output_path), binning_factor=binning_factor)

    assert output_path.exists()
    
    # 1. Assert shape reduction matches expectation
    # Spatial dimensions are the last two dimensions in numpy/tifffile
    expected_spatial_shape = tuple(dim // binning_factor for dim in shape[-2:])
    
    with tifffile.TiffFile(output_path) as tif:
        series_shape = tif.series[0].shape
        # Ensure spatial dimensions (last two) are binned
        assert series_shape[-2:] == expected_spatial_shape
        # Check dtype remains same
        assert tif.series[0].dtype == np.dtype(dtype)

    # 2. Verify metadata update & physical size scaling
    if has_metadata:
        out_meta = read_metadata(str(output_path))
        assert out_meta is not None
        pixels = out_meta.images[0].pixels
        
        # OME size metadata must match output shape
        assert pixels.size_x == expected_spatial_shape[1]
        assert pixels.size_y == expected_spatial_shape[0]

        # Verify physical pixel sizes scaled
        if pixel_size is not None:
            assert pixels.physical_size_x == pytest.approx(pixel_size * binning_factor)
            assert pixels.physical_size_y == pytest.approx(pixel_size * binning_factor)

        # Verify channel names preserved
        if channel_names is not None:
            assert [c.name for c in pixels.channels] == channel_names
