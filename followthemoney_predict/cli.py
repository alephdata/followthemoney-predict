import logging
import sys

import click

from . import data_sources, pipelines, workflow
from .util import FileAnyType


@click.group("predict", help="Utility for FollowTheMoney Predict")
@click.option("--debug", default=False, is_flag=True, envvar="DEBUG")
@click.option("--log", default="-", type=FileAnyType("w"))
@click.pass_context
def cli(ctx, debug, log):
    if ctx.obj is None:
        ctx.obj = {}
    fmt = "%(name)s [%(levelname)s] %(message)s"
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(stream=log, level=level, format=fmt)
    sys.stdout = log
    sys.stderr = log
    logging.debug("Debug Mode")


@cli.group("data")
@click.option("--output-file", required=True, type=FileAnyType("wb"))
@click.option(
    "--data-source",
    default="aleph",
    type=click.Choice(data_sources.DATA_SOURCES.keys()),
)
@click.option("--aleph-api-key", type=str)
@click.option("--aleph-host", type=str)
@click.option("--train-frac", default=0.8, type=click.FloatRange(0, 1))
@click.option(
    "--line-read-limit",
    default=None,
    type=int,
    help="Maximum number of lines to read per source",
)
@click.option(
    "--cache-dir", envvar="FTM_PREDICT_CACHE_DIR", default="/tmp/ftm-predict/"
)
@click.option(
    "--workflow",
    "workflow_type",
    envvar="FTM_PREDICT_WORKFLOW",
    type=click.Choice(workflow.WORKFLOWS),
    default=workflow.WORKFLOWS[0],
)
@click.option("--dask-nworkers", default=1)
@click.option("--dask-threads-per-worker", default=8)
@click.pass_context
def data_cli(
    ctx,
    output_file,
    data_source,
    aleph_host,
    aleph_api_key,
    train_frac,
    cache_dir,
    line_read_limit,
    workflow_type,
    dask_nworkers,
    dask_threads_per_worker,
):
    ctx.obj["output_file"] = output_file
    ctx.obj["data_source_name"] = data_source
    ctx.obj["phases"] = {"train": train_frac, "test": 1 - train_frac}

    ctx.obj["line_read_limit"] = line_read_limit
    ctx.obj["cache_dir"] = cache_dir
    ctx.obj["data_source_type"] = data_source
    ctx.obj["aleph_params"] = {
        "host": aleph_host,
        "api_key": aleph_api_key,
    }
    ctx.obj["data_source"] = data_sources.DATA_SOURCES[data_source](**ctx.obj)

    ctx.obj["workflow_type"] = workflow_type
    ctx.obj["dask_client_kwargs"] = {
        "n_workers": dask_nworkers,
        "threads_per_worker": dask_threads_per_worker,
    }
    ctx.obj["workflow"] = workflow.create_workflow(
        ctx.obj["workflow_type"], ctx.obj["cache_dir"], ctx.obj["dask_client_kwargs"]
    )


@cli.group("model")
@click.option("--output-file", required=True, type=FileAnyType("wb"))
@click.option("--data-file", required=True, type=FileAnyType("rb"))
@click.pass_context
def model_cli(ctx, output_file, data_file):
    ctx.obj["output_file"] = output_file
    ctx.obj["data_file"] = data_file


@cli.group("evaluate")
@click.option("--model-file", required=True, type=FileAnyType("rb"))
@click.option(
    "--cache-dir", envvar="FTM_PREDICT_CACHE_DIR", default="/tmp/ftm-predict/"
)
@click.option(
    "--data-source",
    default="aleph",
    type=click.Choice(data_sources.DATA_SOURCES.keys()),
)
@click.option("--aleph-api-key", type=str, envvar="ALEPHCLIENT_API_KEY")
@click.option("--aleph-host", type=str, envvar="ALEPHCLIENT_HOST")
@click.option(
    "--workflow",
    "workflow_type",
    envvar="FTM_PREDICT_WORKFLOW",
    type=click.Choice(workflow.WORKFLOWS),
    default=workflow.WORKFLOWS[0],
)
@click.option("--dask-nworkers", default=1)
@click.option("--dask-threads-per-worker", default=8)
@click.pass_context
def evaluate_cli(
    ctx,
    model_file,
    cache_dir,
    data_source,
    aleph_api_key,
    aleph_host,
    workflow_type,
    dask_nworkers,
    dask_threads_per_worker,
):
    ctx.obj["model_file"] = model_file

    ctx.obj["cache_dir"] = cache_dir
    ctx.obj["data_source_name"] = data_source
    ctx.obj["aleph_params"] = {
        "host": aleph_host,
        "api_key": aleph_api_key,
    }
    ctx.obj["data_source"] = data_sources.DATA_SOURCES[data_source](**ctx.obj)

    ctx.obj["workflow_type"] = workflow_type
    ctx.obj["dask_client_kwargs"] = {
        "n_workers": dask_nworkers,
        "threads_per_worker": dask_threads_per_worker,
    }
    ctx.obj["workflow"] = workflow.create_workflow(
        ctx.obj["workflow_type"], ctx.obj["cache_dir"], ctx.obj["dask_client_kwargs"]
    )


def main():
    for c in pipelines.DATA_CLI:
        data_cli.add_command(c)
    for c in pipelines.MODEL_CLI:
        model_cli.add_command(c)
    for c in pipelines.EVAULATE_CLI:
        evaluate_cli.add_command(c)
    cli()


if __name__ == "__main__":
    main()
