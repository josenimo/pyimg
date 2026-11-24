"""Memory regression tests for whole-slide-sized images.

dask-image maps each TIFF page to a single dask chunk, so a 48k x 87k channel
plane becomes one ~8 GiB chunk and any computation on it loads the full plane
(and several in parallel). These tests use a plane large enough to tell
"tile-sized reads" from "whole-plane reads" via tracemalloc's peak.
"""
import tracemalloc

import dask
import numpy as np
import pytest
import tifffile

from pyimg.compress import compress_image
from pyimg.utils import describe_image

C, Y, X, TILE = 2, 12288, 12288, 512
PLANE_NBYTES = Y * X * np.dtype(np.uint16).itemsize  # 288 MiB
CHANNEL_NAMES = ["DAPI", "CD45"]


@pytest.fixture(scope="module")
def large_tiled_ome_tiff(tmp_path_factory):
    """Tiled, uncompressed OME-TIFF written tile by tile (no full plane in memory)."""
    path = tmp_path_factory.mktemp("large") / "large.ome.tif"
    tile = np.random.default_rng(0).integers(0, 300, (TILE, TILE), dtype=np.uint16)
    n_tiles = C * (-(-Y // TILE)) * (-(-X // TILE))
    tifffile.imwrite(
        path,
        (tile for _ in range(n_tiles)),
        shape=(C, Y, X),
        dtype=np.uint16,
        tile=(TILE, TILE),
        photometric="minisblack",
        bigtiff=True,
        metadata={"axes": "CYX", "Channel": {"Name": CHANNEL_NAMES}},
    )
    return path


def _peak_bytes(func, *args, **kwargs):
    # Single-threaded dask so the peak reflects chunk size, not the machine's
    # core count (each extra thread holds its own chunk in flight).
    tracemalloc.start()
    try:
        with dask.config.set(scheduler="synchronous"):
            func(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_describe_image_does_not_load_whole_planes(large_tiled_ome_tiff):
    peak = _peak_bytes(describe_image, large_tiled_ome_tiff)
    assert peak < PLANE_NBYTES / 2, f"peak {peak / 2**20:.0f} MiB"


def test_compress_image_does_not_load_whole_planes(large_tiled_ome_tiff, tmp_path):
    out = tmp_path / "large_zlib.ome.tif"
    peak = _peak_bytes(compress_image, str(large_tiled_ome_tiff), str(out), compression="zlib")
    assert peak < PLANE_NBYTES / 2, f"peak {peak / 2**20:.0f} MiB"

    with tifffile.TiffFile(out) as tif:
        # full TiffPage objects (not TiffFrames) so each page's tags can be checked
        pages = [tif.pages.get(i, aspage=True) for i in range(len(tif.pages))]
        assert tif.series[0].axes == "CYX"
        assert tif.series[0].shape == (C, Y, X)
        assert all(page.compression.name == "ADOBE_DEFLATE" for page in pages)
        assert all(page.photometric.name == "MINISBLACK" for page in pages)
        assert pages[0].tile == (TILE, TILE)
        for c in range(C):
            # compare one tile-aligned window per channel against the source
            np.testing.assert_array_equal(
                tif.pages[c].asarray()[:TILE, -TILE:],
                tifffile.imread(large_tiled_ome_tiff, key=c)[:TILE, -TILE:],
            )
    import ome_types
    assert [ch.name for ch in ome_types.from_tiff(out).images[0].pixels.channels] == CHANNEL_NAMES


@pytest.mark.parametrize(
    "shape, photometric, axes",
    [
        ((3, 700, 900), "minisblack", "CYX"),  # fluorescence channels
        ((700, 900, 3), "rgb", "YXS"),  # brightfield RGB (contiguous samples)
        ((700, 900), "minisblack", "YX"),
    ],
)
def test_compress_image_layouts_are_lossless(tmp_path, shape, photometric, axes):
    data = np.random.default_rng(1).integers(0, 255, shape, dtype=np.uint8)
    src = tmp_path / "small.ome.tif"
    tifffile.imwrite(src, data, photometric=photometric, tile=(256, 256), metadata={"axes": axes})
    out = tmp_path / "small_lzw.ome.tif"
    compress_image(str(src), str(out), compression="lzw")

    with tifffile.TiffFile(out) as tif:
        assert tif.series[0].axes == axes
        assert tif.pages[0].photometric.name == photometric.upper()
        np.testing.assert_array_equal(tif.series[0].asarray(), data)
