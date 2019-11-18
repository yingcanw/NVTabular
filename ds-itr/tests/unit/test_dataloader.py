import ds_itr.ds_iterator as ds
import ds_itr.dl_encoder as encoder
import ds_itr.preproc as pp
import cudf
from cudf.tests.utils import assert_eq
import ds_itr.batchloader as bl
import pytest
import torch

import glob
import time
import math
import random
import os


allcols_csv = ["timestamp", "id", "label", "name-string", "x", "y", "z"]
mycols_csv = ["name-string", "id", "label", "x", "y"]
mycols_pq = ["name-cat", "name-string", "id", "label", "x", "y"]
mynames = [
    "Alice",
    "Bob",
    "Charlie",
    "Dan",
    "Edith",
    "Frank",
    "George",
    "Hannah",
    "Ingrid",
    "Jerry",
    "Kevin",
    "Laura",
    "Michael",
    "Norbert",
    "Oliver",
    "Patricia",
    "Quinn",
    "Ray",
    "Sarah",
    "Tim",
    "Ursula",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zelda",
]

sample_stats = {
    "batch_medians": {
        "id": [999.0, 1000.0],
        "x": [-0.051, -0.001],
        "y": [-0.009, -0.001],
    },
    "medians": {"id": 1000.0, "x": -0.001, "y": -0.001},
    "means": {"id": 1000.0, "x": -0.008, "y": -0.001},
    "vars": {"id": 993.65, "x": 0.338, "y": 0.335},
    "stds": {"id": 31.52, "x": 0.581, "y": 0.578},
    "counts": {"id": 4321.0, "x": 4321.0, "y": 4321.0},
    "host_categories": {"name-cat": mynames, "name-string": mynames},
}


@pytest.fixture(scope="session")
def datasets(tmpdir_factory):
    df = cudf.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-04",
        freq="60s",
        dtypes={
            "name-cat": "category",
            "name-string": "category",
            "id": int,
            "label": int,
            "x": float,
            "y": float,
            "z": float,
        },
    ).reset_index()
    df["name-string"] = df["name-string"].astype("O")

    # Add two random null values to each column
    imax = len(df) - 1
    for col in df.columns:
        if col in ["name-cat", "label"]:
            break
        df[col].iloc[random.randint(0, imax)] = None
        df[col].iloc[random.randint(0, imax)] = None

    datadir = tmpdir_factory.mktemp("data")
    datadir = {
        "parquet": tmpdir_factory.mktemp("parquet"),
        "csv": tmpdir_factory.mktemp("csv"),
        "csv-no-header": tmpdir_factory.mktemp("csv-no-header"),
    }

    half = int(len(df) // 2)

    # Write Parquet Dataset
    df.iloc[:half].to_parquet(str(datadir["parquet"]), chunk_size=1000)
    df.iloc[half:].to_parquet(str(datadir["parquet"]), chunk_size=1000)

    # Write CSV Dataset (Leave out categorical column)
    df.iloc[:half].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv"].join("dataset-0.csv")), index=False
    )
    df.iloc[half:].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv"].join("dataset-1.csv")), index=False
    )
    df.iloc[:half].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv-no-header"].join("dataset-0.csv")), header=False, index=False
    )
    df.iloc[half:].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv-no-header"].join("dataset-1.csv")), header=False, index=False
    )

    return datadir


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_file_iterator_ds(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()
    data_itr = bl.FileItrDataset(
        paths[0],
        engine="csv",
        batch_size=batch,
        gpu_memory_frac=0.01,
        columns=mycols_csv,
        names=names,
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_file_iterator_dl(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()
    data_itr = bl.FileItrDataset(
        paths[0],
        engine="csv",
        batch_size=batch,
        gpu_memory_frac=0.01,
        columns=mycols_csv,
        names=names,
    )
    data_chain = torch.utils.data.ChainDataset([data_itr])
    dlc = bl.DLCollator(
        cat_names=["name-string"], cont_names=["x", "y", "id"], label_name=["label"]
    )
    data_dl = bl.DLDataLoader(
        data_itr, collate_fn=dlc.gdf_col, pin_memory=False, num_workers=0
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
def test_gpu_preproc(tmpdir, datasets, dump, gpu_memory_frac, engine):
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

    processor = pp.Preprocessor(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        ops=[pp.FillMissing(), pp.Normalize(), pp.Categorify()],
        to_cpu=True,
    )

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    processor.update_stats(data_itr)
    if dump:
        config_file = tmpdir + "/temp.yaml"
        processor.save_stats(config_file)
        processor.clear_stats()
        processor.load_stats(config_file)

    # Check mean and std
    assert math.isclose(df.x.mean(), processor.means["x"], rel_tol=1e-4)
    assert math.isclose(df.y.mean(), processor.means["y"], rel_tol=1e-4)
    assert math.isclose(df.id.mean(), processor.means["id"], rel_tol=1e-4)
    assert math.isclose(df.x.std(), processor.stds["x"], rel_tol=1e-3)
    assert math.isclose(df.y.std(), processor.stds["y"], rel_tol=1e-3)
    assert math.isclose(df.id.std(), processor.stds["id"], rel_tol=1e-3)

    # Check median (TODO: Improve the accuracy)
    x_median = df.x.dropna().quantile(0.5, interpolation="linear")
    y_median = df.y.dropna().quantile(0.5, interpolation="linear")
    id_median = df.id.dropna().quantile(0.5, interpolation="linear")
    assert math.isclose(x_median, processor.medians["x"], rel_tol=1e1)
    assert math.isclose(y_median, processor.medians["y"], rel_tol=1e1)
    assert math.isclose(id_median, processor.medians["id"], rel_tol=1e-2)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().tolist().sort()
        cats0 = processor.encoders["name-cat"]._cats.keys().to_host().sort()
        assert cats0 == cats_expected0
    cats_expected1 = df["name-string"].unique().tolist().sort()
    cats1 = processor.encoders["name-string"]._cats.keys().to_host().sort()
    assert cats1 == cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(
        tmpdir, data_itr, nfiles=10, shuffle=True, apply_ops=True
    )
    dlc = bl.DLCollator(preproc=processor)
    data_files = [
        bl.FileItrDataset(
            x,
            columns=columns,
            use_row_groups=True,
            gpu_memory_frac=gpu_memory_frac,
            names=allcols_csv,
        )
        for x in glob.glob(str(tmpdir) + "/ds_part.*.parquet")
    ]
    data_itr = torch.utils.data.ChainDataset(data_files)
    dl = bl.DLDataLoader(
        data_itr, collate_fn=dlc.gdf_col, pin_memory=False, num_workers=0
    )

    df_pp = None
    len_df_pp = 0
    for chunk in dl:
        len_df_pp += len(chunk[0][0])

    data_itr = ds.GPUDatasetIterator(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"),
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    x, y = processor.ds_to_tensors(data_itr)

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(
        str(tmpdir) + "/_metadata"
    )
    assert len(x[0]) == len_df_pp
