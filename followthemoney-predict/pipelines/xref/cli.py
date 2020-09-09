import click

from .xref import create_full_stream
from . import data
from . import data_schema
from . import settings


@click.command("xref")
@click.pass_context
def cli(ctx):
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
