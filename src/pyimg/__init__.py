import tifffile
try:
    import tifffile.zarr
    def _dummy_set_partial_values(self, key_start_values, prototype):
        raise NotImplementedError("ZarrTiffStore does not support writes/partial writes.")

    tifffile.zarr.ZarrStore.set_partial_values = _dummy_set_partial_values
    tifffile.zarr.ZarrStore.supports_partial_writes = property(lambda self: False)

    # Clear __abstractmethods__ cache to avoid ABC instantiation error
    if hasattr(tifffile.zarr.ZarrStore, '__abstractmethods__'):
        tifffile.zarr.ZarrStore.__abstractmethods__ = frozenset(
            tifffile.zarr.ZarrStore.__abstractmethods__ - {'set_partial_values', 'supports_partial_writes'}
        )
    if hasattr(tifffile.zarr.ZarrTiffStore, '__abstractmethods__'):
        tifffile.zarr.ZarrTiffStore.__abstractmethods__ = frozenset(
            tifffile.zarr.ZarrTiffStore.__abstractmethods__ - {'set_partial_values', 'supports_partial_writes'}
        )
except (ImportError, AttributeError):
    pass

from .scale_dtype import scale_dtype
from .binning import binning
from .pyramidize import pyramidize

from . import utils