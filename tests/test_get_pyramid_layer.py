import numpy as np
import ome_types
import pytest
import tifffile

from pyimg.get_pyramid_layer import get_pyramid_layer

CHANNEL_NAMES = ["DAPI", "CD3"]
PIXEL_SIZE = 0.5


@pytest.fixture
def ome_pyramid(tmp_path):
    """Two-level, two-channel tiled OME-TIFF pyramid (level 1 = 2x downsampled)."""
    path = tmp_path / "pyramid.ome.tif"
    data = np.random.default_rng(0).integers(0, 300, (2, 1024, 1536), dtype=np.uint16)
    levels = [data, data[:, ::2, ::2]]
    with tifffile.TiffWriter(path, ome=True) as tif:
        tif.write(
            levels[0], photometric="minisblack", tile=(256, 256), subifds=1,
            metadata={
                "axes": "CYX",
                "Channel": {"Name": CHANNEL_NAMES},
                "PhysicalSizeX": PIXEL_SIZE,
                "PhysicalSizeY": PIXEL_SIZE,
            },
        )
        tif.write(levels[1], photometric="minisblack", tile=(256, 256), subfiletype=1)
    return path, levels


@pytest.mark.parametrize("level", [0, 1])
def test_get_pyramid_layer_extracts_level(tmp_path, ome_pyramid, level):
    src, levels = ome_pyramid
    out = tmp_path / f"layer{level}.ome.tif"
    get_pyramid_layer(str(src), str(out), level=level)

    expected = levels[level]
    with tifffile.TiffFile(out) as tif:
        assert tif.series[0].axes == "CYX"
        np.testing.assert_array_equal(tif.series[0].asarray(), expected)

    pixels = ome_types.from_tiff(out).images[0].pixels
    assert (pixels.size_c, pixels.size_y, pixels.size_x) == expected.shape
    assert [ch.name for ch in pixels.channels] == CHANNEL_NAMES
    scale = levels[0].shape[-1] / expected.shape[-1]
    assert pixels.physical_size_x == pytest.approx(PIXEL_SIZE * scale)
    assert pixels.physical_size_y == pytest.approx(PIXEL_SIZE * scale)


def test_get_pyramid_layer_rejects_missing_level(tmp_path, ome_pyramid):
    src, _ = ome_pyramid
    with pytest.raises(ValueError, match="level 2"):
        get_pyramid_layer(str(src), str(tmp_path / "out.ome.tif"), level=2)
