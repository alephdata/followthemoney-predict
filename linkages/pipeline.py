import warnings
import os

from .util import unify_map


USE_DASK = os.environ.get("FTM_PREDICT_USE_DASK", "").lower() == "true"


def init_dask():
    pipeline.Bag.map = unify_map(pipeline.Bag.map, pipeline)
    dask.config.set({"temporary_directory": "/tmp/dask/"})
    cache = Cache(2e9)
    cache.register()
    client = Client(n_workers=1, threads_per_worker=32)
    print(f"Dask Client: {client}")


if USE_DASK:
    try:
        import dask
        import dask.bag as pipeline
        from dask.cache import Cache
        from dask.distributed import Client

        pipeline.init = init_dask
        pipeline.IS_DASK = True
    except ImportError:
        warnings.warn("Dask not found... Using default pipeline", ImportWarning)
        USE_DASK = False

if not USE_DASK:
    from . import dasklike as pipeline  # NOQA

    pipeline.IS_DASK = False
    
