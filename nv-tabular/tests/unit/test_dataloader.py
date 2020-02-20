import nv_tabular.preproc as pp
import nv_tabular.ops as ops
import nv_tabular.ds_iterator as ds
import nv_tabular.batchloader as bl
import cudf
from cudf.tests.utils import assert_eq
import pytest
import torch
from tests.fixtures import *
import shutil


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

    b_size = df_itr.shape[0] // data_itr.gpu_itr.engine.batch_size
    b_size = (
        b_size
        if df_itr.shape[0] % data_itr.gpu_itr.engine.batch_size == 0
        else b_size + 1
    )
    assert b_size == len(data_chain)
    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("preprocessing", [True, False])
def test_gpu_preproc(tmpdir, datasets, dump, gpu_memory_frac, engine, preprocessing):
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


    processor = pp.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        to_cpu=True,
    )
    
    processor.add_feature([ops.FillMissing(), ops.LogOp(preprocessing=preprocessing)])
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify())
    processor.finalize()

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

    def get_norms(tar: cudf.Series):
        ser_median = tar.dropna().quantile(0.5, interpolation="linear")
        gdf = tar.fillna(ser_median)
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Zerofill, Log

    assert math.isclose(
        get_norms(df.x).mean(), processor.stats["means"]["x_FillMissing_LogOp"], rel_tol=1e-2
    )
    assert math.isclose(
        get_norms(df.y).mean(), processor.stats["means"]["y_FillMissing_LogOp"], rel_tol=1e-2
    )
    #     assert math.isclose(get_norms(df.id).mean(), processor.stats["means"]["id_FillMissing"], rel_tol=1e-4)
    assert math.isclose(
        get_norms(df.x).std(), processor.stats["stds"]["x_FillMissing_LogOp"], rel_tol=1e-2
    )
    assert math.isclose(
        get_norms(df.y).std(), processor.stats["stds"]["y_FillMissing_LogOp"], rel_tol=1e-2
    )
    #     assert math.isclose(get_norms(df.id).std(), processor.stats["stds"]["id_FillMissing"], rel_tol=1e-3)

    # Check median (TODO: Improve the accuracy)
    x_median = df.x.dropna().quantile(0.5, interpolation="linear")
    y_median = df.y.dropna().quantile(0.5, interpolation="linear")
    id_median = df.id.dropna().quantile(0.5, interpolation="linear")
    assert math.isclose(x_median, processor.stats["medians"]["x"], rel_tol=1e1)
    assert math.isclose(y_median, processor.stats["medians"]["y"], rel_tol=1e1)
    assert math.isclose(id_median, processor.stats["medians"]["id"], rel_tol=1e-2)


    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = processor.stats["encoders"]["name-cat"]._cats.values_to_string()
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = processor.stats["encoders"]["name-string"]._cats.values_to_string()
    assert cats1 == ["None"] + cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(
        tmpdir, data_itr, nfiles=10, shuffle=True, apply_ops=True
    )
    
    processor.create_final_cols()
    
    #if preprocessing
    if not preprocessing:
        for col in cont_names:
            assert f"{col}_FillMissing_LogOp" in processor.columns_ctx["final"]["cols"]["continuous"]

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

    x = processor.ds_to_tensors(data_itr)

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(
        str(tmpdir) + "/_metadata"
    )
    assert len(x[0]) == len_df_pp

    itr_ds = bl.TensorItrDataset([x[0], x[1], x[2]], batch_size=512000)
    count_tens_itr = 0
    for data_gd in itr_ds:
        count_tens_itr += len(data_gd[1])
        assert data_gd[0][0].shape[1] > 0
        assert data_gd[0][1].shape[1] > 0

    assert len_df_pp == count_tens_itr
    if os.path.exists(processor.ds_exports):
        shutil.rmtree(processor.ds_exports)
