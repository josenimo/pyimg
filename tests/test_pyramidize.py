import numpy as np
import tifffile

from pyimg.pyramidize import pyramidize
from tests.helpers import create_test_image


def _levels(path):
    with tifffile.TiffFile(path) as tif:
        return [level.asarray() for level in tif.series[0].levels]


def test_pyramidize_single_page_multichannel(tmp_path):
    """A CYX array stored as one planar page is read by dask-image as (1, C, Y, X)."""
    src = create_test_image(
        tmp_path / "rgb.ome.tif", (3, 1200, 1200), np.uint16, "CYX",
        channel_names=["a", "b", "c"],
    )
    out = tmp_path / "rgb_pyramid.ome.tif"
    pyramidize(str(src), str(out), min_size=200)

    levels = _levels(out)
    assert [lvl.shape for lvl in levels] == [(3, 1200, 1200), (3, 400, 400), (3, 133, 133)]
    np.testing.assert_array_equal(levels[0], tifffile.imread(src))

