import ds_itr.ds_iterator as ds
import ds_itr.dl_encoder as encoder
import ds_itr.preproc as pp
import cudf
import numpy as np
from cudf.tests.utils import assert_eq
import pytest

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
def test_gpu_file_iterator_parquet(datasets, batch):
    paths = glob.glob(str(datasets["parquet"]) + "/*.parquet")
    df_expect = cudf.read_parquet(paths[0], columns=mycols_pq)
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUFileIterator(
        paths[0], batch_size=batch, gpu_memory_frac=0.01, columns=mycols_pq
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


def test_gpu_file_iterator_parquet_row_groups(datasets):
    paths = glob.glob(str(datasets["parquet"]) + "/*.parquet")
    df_expect = cudf.read_parquet(paths[0], columns=mycols_pq)
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUFileIterator(
        paths[0], use_row_groups=True, gpu_memory_frac=0.0, columns=mycols_pq
    )
    for idx, data_gd in enumerate(data_itr):
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    # Make sure the iteration count matches the row-group count
    (_, num_row_groups, _) = cudf.io.read_parquet_metadata(paths[0])
    assert num_row_groups == (idx + 1)

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_file_iterator_csv(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUFileIterator(
        paths[0],
        batch_size=batch,
        gpu_memory_frac=0.01,
        columns=mycols_csv,
        names=names,
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
def test_gpu_dataset_iterator_parquet(datasets, batch):
    paths = glob.glob(str(datasets["parquet"]) + "/*.parquet")
    df_expect = cudf.read_parquet(paths[0], columns=mycols_pq)
    df_expect = cudf.concat(
        [df_expect, cudf.read_parquet(paths[1], columns=mycols_pq)], axis=0
    )
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUDatasetIterator(
        paths, batch_size=batch, gpu_memory_frac=0.01, columns=mycols_pq
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_dataset_iterator_csv(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    df_expect1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    df_expect2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df_expect = cudf.concat([df_expect1, df_expect2], axis=0)
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUDatasetIterator(
        paths,
        batch_size=batch,
        gpu_memory_frac=0.01,
        columns=mycols_csv,
        names=allcols_csv,
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

    data_itr_2 = ds.GPUDatasetIterator(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"),
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
    )

    df_pp = None
    for chunk in data_itr_2:
        df_pp = cudf.concat([df_pp, chunk], axis=0) if df_pp else chunk

    if engine == "parquet":
        assert df_pp["name-cat"].dtype == "int64"
    assert df_pp["name-string"].dtype == "int64"

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(
        str(tmpdir) + "/_metadata"
    )
    assert num_rows == len(df_pp)


def test_pq_to_pq_processed(tmpdir, datasets):
    indir = str(datasets["parquet"])
    outdir = str(tmpdir)
    cat_names = ["name-cat", "name-string"]
    columns = mycols_pq
    cont_names = ["x", "y", "id"]
    label_name = ["label"]
    chunk_size = 100

    processor = pp.Preprocessor(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        ops=[pp.FillMissing(), pp.Normalize(), pp.Categorify()],
        to_cpu=True,
    )

    processor.load_stats(sample_stats)
    processor.pq_to_pq_processed(
        indir,
        outdir,
        columns=mycols_pq,
        shuffle=True,
        apply_ops=True,
        chunk_size=chunk_size,
    )

    # TODO: Test that the new parquet dataset is processed correctly
    old_paths = glob.glob(indir + "/*.parquet")
    new_paths = glob.glob(outdir + "/*.parquet")
    assert len(new_paths) == len(old_paths)

    meta = cudf.io.read_parquet_metadata(outdir + "/_metadata")
    assert meta[2] == mycols_pq
    assert meta[0] // meta[1] <= chunk_size


def test_estimated_row_size(tmpdir):
    # Make sure the row_size estimate is what we expect...
    size = 1000
    df = cudf.DataFrame(
        {
            "int32": np.arange(size, dtype="int32"),
            "int64": np.arange(size, dtype="int64"),
            "float64": np.arange(size, dtype="float64"),
            "str": np.random.choice(["cat", "bat", "dog"], size=size),
        }
    )

    # Write parquet File
    fn_csv = str(tmpdir) + "/temp.csv"
    df.to_csv(fn_csv, index=False)

    # Write csv File
    df.to_parquet(str(tmpdir))
    fn_pq = glob.glob(str(tmpdir) + "/*.parquet")[0]

    # Use PyArrow to get "accurate" in-memory row size
    read_byte_size = 0
    for col in cudf.read_parquet(fn_pq)._columns:
        if col.dtype == "object":
            max_size = len(max(col)) // 2
            read_byte_size += int(max_size)
            import pdb; pdb.set_trace()
        else:
            read_byte_size += col.dtype.itemsize

    # Check parquet estimate
    reader_pq = ds.PQFileReader(fn_pq, 0.1, None, row_size=None)
    estimated_row_size_pq = reader_pq.estimated_row_size
    assert estimated_row_size_pq == read_byte_size

    # Check csv estimate
    reader_csv = ds.CSVFileReader(
        fn_csv, 0.1, None, row_size=None, dtype=["int32", "int64", "float64", "str"]
    )
    estimated_row_size_csv = reader_csv.estimated_row_size
    assert estimated_row_size_csv == read_byte_size
