from functools import singledispatch
from typing import Tuple

from followthemoney import model
from followthemoney.proxy import EntityProxy
from followthemoney.schema import Schema


import torch
from torch import Tensor
from torch import jit
from torch.nn.utils.rnn import PackedSequence


@jit.script
def sorted_lengths(pack: PackedSequence) -> Tuple[Tensor, Tensor]:
    indices = torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )
    lengths = ((indices + 1)[:, None] <= pack.batch_sizes[None, :]).long().sum(dim=1)
    return lengths, indices


@jit.script
def sorted_first_indices(pack: PackedSequence) -> Tensor:
    print(pack.batch_sizes.device)
    return torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )


@jit.script
def sorted_last_indices(pack: PackedSequence) -> Tensor:
    lengths, indices = sorted_lengths(pack)
    cum_batch_sizes = torch.cat(
        [
            pack.batch_sizes.new_zeros((2,)),
            torch.cumsum(pack.batch_sizes, dim=0),
        ],
        dim=0,
    )
    return cum_batch_sizes[lengths] + indices


@jit.script
def first_items(pack: PackedSequence, unsort: bool) -> Tensor:
    if unsort and pack.unsorted_indices is not None:
        return pack.data[pack.unsorted_indices]
    else:
        return pack.data[: pack.batch_sizes[0]]


@jit.script
def last_items(pack: PackedSequence, unsort: bool) -> Tensor:
    indices = sorted_last_indices(pack=pack)
    if unsort and pack.unsorted_indices is not None:
        indices = indices[pack.unsorted_indices]
    return pack.data[indices]


@singledispatch
def filter_schema(schema):
    raise ValueError


@filter_schema.register(dict)
def _(entity):
    schema = model.get(entity["schema"])
    return filter_schema(schema)


@filter_schema.register(EntityProxy)
def _(proxy):
    return filter_schema(proxy.schema)


@filter_schema.register(Schema)
def _(schema):
    return schema.is_a(model.get("Thing")) and not schema.is_a(model.get("Document"))
