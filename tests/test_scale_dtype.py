import pytest
import numpy as np
import tifffile
import ome_types
from pathlib import Path

from pyimg.scale_dtype import scale_dtype
from pyimg.utils.read_metadata import read_metadata
from .helpers import create_test_image

@pytest.mark.parametrize(
    "shape,layout,dtype,has_metadata,channel_names,pixel_size",
    [
        # XY (2D), standard uint16, with metadata (custom channels/pixelsize)
        ((50, 60), "XY", np.uint16, True, ["Gray"], 0.8),
        # XY (2D), standard uint16, without metadata
        ((50, 60), "XY", np.uint16, False, None, None),
        # CYX (3D), standard uint16, with metadata
        ((3, 50, 60), "CYX", np.uint16, True, ["DAPI", "FITC", "Cy5"], 1.2),
        # CYX (3D), standard uint16, without metadata
        ((3, 50, 60), "CYX", np.uint16, False, None, None),
        # CXY (3D), standard uint16, with metadata
        ((3, 60, 50), "CXY", np.uint16, True, ["Ch1", "Ch2", "Ch3"], 1.5),
        # CYX (3D), float32, with metadata
        ((2, 40, 40), "CYX", np.float32, True, ["Red", "Green"], 0.5),
        # CYX (3D), uint8, with metadata
        ((4, 30, 30), "CYX", np.uint8, True, None, 2.0),
    ]
)
def test_image_creation_and_scaling(
    tmp_path, shape, layout, dtype, has_metadata, channel_names, pixel_size
):
    """Test creating and scaling images across all common layouts and metadata options."""
    input_path = tmp_path / f"test_img_{layout}_{dtype.__name__ if hasattr(dtype, '__name__') else str(dtype)}.tif"
    output_path = tmp_path / "scaled_output.tif"

    # Create the test image
    create_test_image(
        path=input_path,
        shape=shape,
        dtype=dtype,
        layout=layout,
        has_metadata=has_metadata,
        channel_names=channel_names,
        pixel_size=pixel_size
    )

    # 1. Assert input file properties
    assert input_path.exists()
    with tifffile.TiffFile(input_path) as tif:
        assert tif.series[0].dtype == np.dtype(dtype)

    if has_metadata:
        in_meta = read_metadata(str(input_path))
        assert in_meta is not None
        pixels = in_meta.images[0].pixels
        
        # Verify physical pixel sizes if specified
        if pixel_size is not None:
            assert pixels.physical_size_x == pytest.approx(pixel_size)
            assert pixels.physical_size_y == pytest.approx(pixel_size)

        # Verify channel names
        if channel_names is not None:
            assert [c.name for c in pixels.channels] == channel_names

    # 2. Run scaling to uint8 (expected to fail on certain shapes/layouts until fixed)
    # Wrap in try-except or let fail to see failures clearly.
    # We will let it fail so we can diagnose.
    scale_dtype(str(input_path), str(output_path), output_dtype="uint8")

    # 3. Assert output file properties
    assert output_path.exists()
    with tifffile.TiffFile(output_path) as tif:
        assert tif.series[0].dtype == np.uint8

    # 4. Assert metadata preservation
    if has_metadata:
        out_meta = read_metadata(str(output_path))
        assert out_meta is not None
        pixels_out = out_meta.images[0].pixels

        # Dtype should be updated to uint8 in the metadata
        assert pixels_out.type.value == "uint8"

        # Pixel size and channel names should be preserved
        if pixel_size is not None:
            assert pixels_out.physical_size_x == pytest.approx(pixel_size)
            assert pixels_out.physical_size_y == pytest.approx(pixel_size)
        if channel_names is not None:
            assert [c.name for c in pixels_out.channels] == channel_names