import itertools as IT
import logging
import pickle

import click
import pandas as pd

from . import data, data_schema, settings
from .models import XrefModel, XrefLinear, XrefXGBoost
from .models.util import format_prediction
from .pipeline import create_dataframe_from_entities, create_full_stream


@click.command("xref")
@click.pass_context
def data_cli(ctx):
    schema = data_schema.create_schema(
        ctx.obj["phases"].keys(), ["profile", "negative", "positive"]
    )

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
@click.option("--best-of", type=int, default=1)
@click.pass_context
def xref_models(ctx, best_of):
    ctx.obj["best_of"] = best_of


def train_model(ctx, model_class):
    model_best = None
    best_of = ctx.obj["best_of"]
    for i in range(best_of):
        model = model_class()
        logging.info(f"Training model {i+1} out of {best_of}: {model}")
        model.fit_parquet(ctx.obj["data_file"])
        if model_best is not None:
            logging.info(f"Best Model: {model_best.meta['scores']}")
        logging.info(f"Current Model: {model.meta['scores']}")
        if model_best is None or model.better_than(model_best, metric="accuracy"):
            logging.info("Keeping new model")
            model_best = model
    ctx.obj["output_file"].write(model_best.dumps())


@xref_models.command("xgboost")
@click.pass_context
def xgboost_cli(ctx):
    train_model(ctx, XrefXGBoost)


@xref_models.command("linear")
@click.pass_context
def linear_cli(ctx):
    train_model(ctx, XrefLinear)


@click.command("xref")
@click.option("--entity", "-e", "entity_ids", required=True, multiple=True, type=str)
@click.option(
    "--collection-foreign-id", "-c", "collection_fids", multiple=True, type=str
)
@click.option("--summary", "-s", is_flag=True, default=False)
@click.pass_context
def evaluate_cli(ctx, entity_ids, collection_fids, summary):
    model = XrefModel.loads(ctx.obj["model_file"].read())
    logging.info(f"Evaluating using model: {model}")

    workflow = ctx.obj["workflow"]
    data_source = ctx.obj["data_source"]
    schema = data_schema.create_schema(["evaluate"], ["evaluate"])

    if entity_ids and not hasattr(data_source, "get_entity"):
        raise click.BadParameter(
            "Specifying entities requires a data source that supports fetching individual entities"
        )

    collections_entities = (
        data_source.get_collection_entities_by_foreign_id(fid, settings.SCHEMAS)
        for fid in collection_fids
    )
    entities = (data_source.get_entity(eid) for eid in entity_ids)

    if entity_ids and not collection_fids:
        pairs = IT.combinations(entities, 2)
    elif entity_ids and collection_fids:
        pairs = (
            (e, ce)
            for e in entities
            for collection_entities in collections_entities
            for ce in collection_entities
        )
    else:
        pairs = IT.combinations(IT.chain.from_iterable(collections_entities), 2)

    df = create_dataframe_from_entities(
        workflow.from_sequence(pairs), meta=schema, source="evaluate"
    )
    y_predict_proba = model.predict(df)

    if summary:
        model.describe_predictions(df, y_predict_proba)
    else:
        for (idx, sample), prediction in zip(df.iterrows(), y_predict_proba):
            print(format_prediction(sample, prediction))
