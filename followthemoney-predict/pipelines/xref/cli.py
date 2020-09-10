import pickle

import click
import pandas as pd

from .xref import create_full_stream
from .models import fit_xgboost, fit_linear
from . import data
from . import data_schema
from . import settings


@click.command("xref")
@click.pass_context
def data_cli(ctx):
    schema = data_schema.create_schema(ctx.obj["phases"])

    stream_set = data.get_data_streams(
        ctx.obj["workflow"], ctx.obj["data_source"], settings.NEGATIVE_COLLECTION_FIDS
    )
    pairs = create_full_stream(
        ctx.obj["workflow"],
        stream_set,
        n_lines_read=ctx.obj["line_read_limit"],
        phases=ctx.obj["phases"],
    )

    if ctx.obj["workflow"].IS_DASK:
        pairs = create_full_stream(stream_set)
        df = pairs.to_dataframe(meta=schema)
        df.to_parquet(ctx.obj["output_file"], schema="infer")
    else:
        pairs.to_parquet(ctx.obj["output_file"], meta=schema)


@click.group("xref")
@click.pass_context
def xref_models(ctx):
    pass


@xref_models.command("xgboost")
@click.pass_context
def xgboost_cli(ctx):
    df = pd.read_parquet(ctx.obj["data_file"]).reset_index()
    clf, scores = fit_xgboost(df)
    model_spec = {
        "meta": {
            "type": "xgboost",
            "version": 0.1,
            "training_data": ctx.obj["data_file"].name,
            "scores": scores,
        },
        "model": clf,
    }
    pickle.dump(model_spec, ctx.obj["output_file"])


@xref_models.command("linear")
@click.pass_context
def linear_cli(ctx):
    df = pd.read_parquet(ctx.obj["data_file"]).reset_index()
    clf, scores = fit_linear(df)
    model_spec = {
        "meta": {
            "type": "linear",
            "version": 0.1,
            "training_data": ctx.obj["data_file"].name,
            "scores": scores,
        },
        "model": clf,
    }
    pickle.dump(model_spec, ctx.obj["output_file"])
