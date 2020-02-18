import nv_tabular.preproc as pp
import nv_tabular.ops as ops
import nv_tabular.dl_encoder as encoder
import nv_tabular.ds_iterator as ds
import nv_tabular.batchloader as bl
import cudf
import numpy as np
from cudf.tests.utils import assert_eq
import pytest
from tests.fixtures import *

import glob
import time
import math
import random
import os
import shutil


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
def test_minmax(tmpdir, datasets, gpu_memory_frac, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    config = pp.get_new_config()
    config["PP"]["all"] = [ops.MinMax()]

    processor = pp.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        to_cpu=False,
    )

    processor.update_stats(data_itr)

    x_min = min(df["x"])
    y_min = min(df["y"])
    name_min = min(df["name-string"])
    assert x_min == processor.stats["mins"]["x"]
    assert y_min == processor.stats["mins"]["y"]
    assert name_min == processor.stats["mins"]["name-string"]
    x_max = max(df["x"])
    y_max = max(df["y"])
    name_max = max(df["name-string"])
    assert x_max == processor.stats["maxs"]["x"]
    assert y_max == processor.stats["maxs"]["y"]
    assert name_max == processor.stats["maxs"]["name-string"]
    if os.path.exists(processor.ds_exports):
        shutil.rmtree(processor.ds_exports)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
def test_moments(tmpdir, datasets, gpu_memory_frac, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    config = pp.get_new_config()
    config["PP"]["continuous"] = [ops.Moments()]

    processor = pp.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        to_cpu=False,
    )

    processor.update_stats(data_itr)

    # Check mean and std
    assert math.isclose(df.x.mean(), processor.stats["means"]["x"], rel_tol=1e-4)
    assert math.isclose(df.y.mean(), processor.stats["means"]["y"], rel_tol=1e-4)
    assert math.isclose(df.id.mean(), processor.stats["means"]["id"], rel_tol=1e-4)
    assert math.isclose(df.x.std(), processor.stats["stds"]["x"], rel_tol=1e-3)
    assert math.isclose(df.y.std(), processor.stats["stds"]["y"], rel_tol=1e-3)
    assert math.isclose(df.id.std(), processor.stats["stds"]["id"], rel_tol=1e-3)
    if os.path.exists(processor.ds_exports):
        shutil.rmtree(processor.ds_exports)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
def test_encoder(tmpdir, datasets, gpu_memory_frac, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    config = pp.get_new_config()
    config["PP"]["categorical"] = [ops.Encoder()]

    processor = pp.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        to_cpu=False,
    )

    processor.update_stats(data_itr)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = processor.stats["encoders"]["name-cat"]._cats.values_to_string()
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = processor.stats["encoders"]["name-string"]._cats.values_to_string()
    assert cats1 == ["None"] + cats_expected1
    if os.path.exists(processor.ds_exports):
        shutil.rmtree(processor.ds_exports)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
def test_median(tmpdir, datasets, gpu_memory_frac, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    config = pp.get_new_config()
    config["PP"]["continuous"] = [ops.Median()]

    processor = pp.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        to_cpu=False,
    )

    processor.update_stats(data_itr)

    # Check median (TODO: Improve the accuracy)
    x_median = df.x.dropna().quantile(0.5, interpolation="linear")
    y_median = df.y.dropna().quantile(0.5, interpolation="linear")
    id_median = df.id.dropna().quantile(0.5, interpolation="linear")
    assert math.isclose(x_median, processor.stats["medians"]["x"], rel_tol=1e1)
    assert math.isclose(y_median, processor.stats["medians"]["y"], rel_tol=1e1)
    assert math.isclose(id_median, processor.stats["medians"]["id"], rel_tol=1e-2)
    if os.path.exists(processor.ds_exports):
        shutil.rmtree(processor.ds_exports)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
def test_log(tmpdir, datasets, gpu_memory_frac, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    log_op = ops.LogOp()

    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = cont_names

    for gdf in data_itr:
        new_gdf = log_op.apply_op(gdf, columns_ctx, "continuous")
        assert new_gdf[cont_names] == np.log(gdf[cont_names].astype(np.float32))
