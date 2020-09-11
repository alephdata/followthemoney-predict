import os
import warnings

from .util import unify_map

WORKFLOWS = ["native", "dask"]


def init_dask(workflow, cache_dir=None, client_kwargs=None):
    import dask
    from dask.cache import Cache
    from dask.distributed import Client

    workflow.Bag.map = unify_map(workflow.Bag.map, workflow)
    if cache_dir:
        dask.config.set({"temporary_directory": cache_dir})

    cache = Cache(2e9)
    cache.register()
    client = Client(client_kwargs)
    print(f"Dask Client: {client}")
    return workflow


def create_workflow(workflow_type, cache_dir, dask_client_kwargs=None):
    if workflow_type == "dask":
        import dask.bag as workflow

        workflow.init = init_dask
        workflow.IS_DASK = True
        init_dask(workflow, cache_dir, dask_client_kwargs)
    else:
        from . import dasklike as workflow  # NOQA

        workflow.IS_DASK = False

    return workflow
