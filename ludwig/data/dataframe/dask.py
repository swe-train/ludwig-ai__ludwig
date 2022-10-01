#! /usr/bin/env python
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
from typing import Dict

import dask
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.utils.data_utils import get_pa_schema, split_by_slices
from ludwig.utils.dataframe_utils import set_index_name

TMP_COLUMN = "__TMP_COLUMN__"


logger = logging.getLogger(__name__)


def from_pandas_refs(dfs):
    """Create a dataset from a list of Ray object references to Pandas dataframes.

    Args:
        dfs: A Ray object references to pandas dataframe, or a list of
             Ray object references to pandas dataframes.

    Returns:
        Dataset holding Arrow records read from the dataframes.
    """
    import ray
    from ray.data._internal.block_list import BlockList
    from ray.data._internal.plan import ExecutionPlan
    from ray.data._internal.remote_fn import cached_remote_fn
    from ray.data._internal.stats import DatasetStats
    from ray.data.context import DatasetContext
    from ray.data.dataset import Dataset
    from ray.data.read_api import _df_to_block, _get_metadata
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if isinstance(dfs, ray.ObjectRef):
        dfs = [dfs]
    elif isinstance(dfs, list):
        for df in dfs:
            if not isinstance(df, ray.ObjectRef):
                raise ValueError("Expected list of Ray object refs, " f"got list containing {type(df)}")
    else:
        raise ValueError("Expected Ray object ref or list of Ray object refs, " f"got {type(df)}")

    scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=ray.util.get_current_placement_group())

    context = DatasetContext.get_current()
    if context.enable_pandas_block:
        get_metadata = cached_remote_fn(_get_metadata, scheduling_strategy=scheduling_strategy)
        metadata = ray.get([get_metadata.remote(df) for df in dfs])
        return Dataset(
            ExecutionPlan(
                BlockList(dfs, metadata, owned_by_consumer=False),
                DatasetStats(stages={"from_pandas_refs": metadata}, parent=None),
                run_by_consumer=False,
            ),
            0,
            False,
        )

    df_to_block = cached_remote_fn(_df_to_block, num_returns=2, scheduling_strategy=scheduling_strategy)

    res = [df_to_block.remote(df) for df in dfs]
    blocks, metadata = map(list, zip(*res))
    metadata = ray.get(metadata)
    return Dataset(
        ExecutionPlan(
            BlockList(blocks, metadata, owned_by_consumer=False),
            DatasetStats(stages={"from_pandas_refs": metadata}, parent=None),
            run_by_consumer=False,
        ),
        0,
        False,
    )


def from_dask(df: "dask.DataFrame"):
    """Create a dataset from a Dask DataFrame.

    Args:
        df: A Dask DataFrame.

    Returns:
        Dataset holding Arrow records read from the DataFrame.
    """
    import dask
    import ray
    from ray.util.dask import ray_dask_get

    partitions = df.to_delayed()
    persisted_partitions = dask.persist(*partitions, scheduler=ray_dask_get, optimize_graph=False)

    import pandas

    def to_ref(df):
        if isinstance(df, pandas.DataFrame):
            return ray.put(df)
        elif isinstance(df, ray.ObjectRef):
            return df
        else:
            raise ValueError("Expected a Ray object ref or a Pandas DataFrame, " f"got {type(df)}")

    print("!!! FROM PANDAS REFS")
    return from_pandas_refs([to_ref(next(iter(part.dask.values()))) for part in persisted_partitions])


def set_scheduler(scheduler):
    dask.config.set(scheduler=scheduler)


def reset_index_across_all_partitions(df):
    """Compute a monotonically increasing index across all partitions.

    This differs from dd.reset_index, which computes an independent index for each partition.
    Source: https://stackoverflow.com/questions/61395351/how-to-reset-index-on-concatenated-dataframe-in-dask
    """
    # Create temporary column of ones
    df = df.assign(**{TMP_COLUMN: 1})

    # Set the index to the cumulative sum of TMP_COLUMN, which we know to be sorted; this improves efficiency.
    df = df.set_index(df[TMP_COLUMN].cumsum() - 1, sorted=True)

    # Drop temporary column and ensure the index is not named TMP_COLUMN
    df = df.drop(columns=TMP_COLUMN)
    df = df.map_partitions(lambda pd_df: set_index_name(pd_df, None))
    return df


class DaskEngine(DataFrameEngine):
    def __init__(self, parallelism=None, persist=True, _use_ray=True, **kwargs):
        from ray.util.dask import ray_dask_get

        self._parallelism = parallelism
        self._persist = persist
        if _use_ray:
            set_scheduler(ray_dask_get)

    def set_parallelism(self, parallelism):
        self._parallelism = parallelism

    def df_like(self, df: dd.DataFrame, proc_cols: Dict[str, dd.Series]):
        """Outer joins the given DataFrame with the given processed columns.

        NOTE: If any of the processed columns have been repartitioned, the original index is replaced with a
        monotonically increasing index, which is used to define the new divisions and align the various partitions.
        """
        # Our goal is to preserve the index of the input dataframe but to drop
        # all its columns. Because to_frame() creates a column from the index,
        # we need to drop it immediately following creation.
        dataset = df.index.to_frame(name=TMP_COLUMN).drop(columns=TMP_COLUMN)

        repartitioned_cols = {}
        for k, v in proc_cols.items():
            if v.npartitions == dataset.npartitions:
                # Outer join cols with equal partitions
                v.divisions = dataset.divisions
                dataset[k] = v
            else:
                # If partitions have changed (e.g. due to conversion from Ray dataset), we handle separately
                repartitioned_cols[k] = v

        # Assumes that there is a globally unique index (see preprocessing.build_dataset)
        if repartitioned_cols:
            if not dataset.known_divisions:
                # Sometimes divisions are unknown despite having a usable index– set_index to know divisions
                dataset = dataset.assign(**{TMP_COLUMN: dataset.index})
                dataset = dataset.set_index(TMP_COLUMN, drop=True)
                dataset = dataset.map_partitions(lambda pd_df: set_index_name(pd_df, dataset.index.name))

            # Find the divisions of the column with the largest number of partitions
            proc_col_with_max_npartitions = max(repartitioned_cols.values(), key=lambda x: x.npartitions)
            new_divisions = proc_col_with_max_npartitions.divisions

            # Repartition all columns to have the same divisions
            dataset = dataset.repartition(new_divisions)
            repartitioned_cols = {k: v.repartition(new_divisions) for k, v in repartitioned_cols.items()}

            # Outer join the remaining columns
            for k, v in repartitioned_cols.items():
                dataset[k] = v

        return dataset

    def parallelize(self, data):
        if self.parallelism:
            return data.repartition(self.parallelism)
        return data

    def persist(self, data):
        # No graph optimizations to prevent dropping custom annotations
        # https://github.com/dask/dask/issues/7036
        return data.persist(optimize_graph=False) if self._persist else data

    def concat(self, dfs):
        return self.df_lib.multi.concat(dfs)

    def compute(self, data):
        return data.compute()

    def from_pandas(self, df):
        parallelism = self._parallelism or 1
        return dd.from_pandas(df, npartitions=parallelism)

    def map_objects(self, series, map_fn, meta=None):
        meta = meta if meta is not None else ("data", "object")
        return series.map(map_fn, meta=meta)

    def map_partitions(self, series, map_fn, meta=None):
        meta = meta if meta is not None else ("data", "object")
        return series.map_partitions(map_fn, meta=meta)

    def map_batches(self, series, map_fn):
        ds = from_dask(series)
        ds = ds.map_batches(map_fn, batch_format="pandas")
        return ds.to_dask()

    def apply_objects(self, df, apply_fn, meta=None):
        meta = meta if meta is not None else ("data", "object")
        return df.apply(apply_fn, axis=1, meta=meta)

    def reduce_objects(self, series, reduce_fn):
        return series.reduction(reduce_fn, aggregate=reduce_fn, meta=("data", "object")).compute()[0]

    def split(self, df, probabilities):
        # Split the DataFrame proprotionately along partitions. This is an inexact solution designed
        # to speed up the split process, as splitting within partitions would be significantly
        # more expensive.
        # TODO(travis): revisit in the future to make this more precise

        # First ensure that every split receives at least one partition.
        # If not, we need to increase the number of partitions to satisfy this constraint.
        min_prob = min(probabilities)
        min_partitions = int(1 / min_prob)
        if df.npartitions < min_partitions:
            df = df.repartition(min_partitions)

        n = df.npartitions
        slices = df.partitions
        return split_by_slices(slices, n, probabilities)

    def remove_empty_partitions(self, df):
        # Reference: https://stackoverflow.com/questions/47812785/remove-empty-partitions-in-dask
        ll = list(df.map_partitions(len).compute())
        if all([ll_i > 0 for ll_i in ll]):
            return df

        df_delayed = df.to_delayed()
        df_delayed_new = list()
        empty_partition = None
        for ix, n in enumerate(ll):
            if n == 0:
                empty_partition = df.get_partition(ix)
            else:
                df_delayed_new.append(df_delayed[ix])
        df = dd.from_delayed(df_delayed_new, meta=empty_partition)
        return df

    def to_parquet(self, df, path, index=False):
        schema = get_pa_schema(df)
        with ProgressBar():
            df.to_parquet(
                path,
                engine="pyarrow",
                write_index=index,
                schema=schema,
            )

    def to_ray_dataset(self, df):
        return from_dask(df)

    def from_ray_dataset(self, dataset) -> dd.DataFrame:
        return dataset.to_dask()

    def reset_index(self, df):
        return reset_index_across_all_partitions(df)

    @property
    def array_lib(self):
        return da

    @property
    def df_lib(self):
        return dd

    @property
    def parallelism(self):
        return self._parallelism

    @property
    def partitioned(self):
        return True
