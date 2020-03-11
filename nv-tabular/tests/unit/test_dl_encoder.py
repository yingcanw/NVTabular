import nv_tabular.dl_encoder as encoder
import nv_tabular.ds_iterator as ds
import cudf
import pytest
import torch
from tests.fixtures import *
import os


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("use_frequency", [True, False])
def test_dl_encoder_fit_transform_fim(datasets, batch, dskey, use_frequency):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    # create file iterator to go through the
    enc = encoder.DLLabelEncoder("name-string")
    if use_frequency:
        enc.fit_freq(df_expect["name-string"])
        enc.fit_freq_finalize()
        new_ser = enc.transform_freq(df_expect["name-string"])
    else:
        enc.fit_unique(df_expect["name-string"])
        new_ser = enc.transform_unique(df_expect["name-string"])
    unis = set(df_expect["name-string"])
    assert len(unis) == max(new_ser)
    for file in enc.file_paths:
        os.remove(file)


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("use_frequency", [True, False])
def test_dl_encoder_fit_transform_ltm(datasets, batch, dskey, use_frequency):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    data_itr = ds.GPUDatasetIterator(
        paths[0], batch_size=batch, gpu_memory_frac=2e-8, names=names
    )
    enc = encoder.DLLabelEncoder(
        "name-string", path=str(datasets["cats"]), limit_frac=1e-10, use_frequency=use_frequency
    )
    for chunk in data_itr:
        enc.fit(chunk["name-string"])
  
    if use_frequency:
        enc.fit_freq_finalize()
    new_ser = enc.transform(df_expect["name-string"])
    unis = df_expect["name-string"].unique().values_to_string()
    # set does not pick up None values so must be added if found in
    cat_count = max(new_ser)
    assert len(unis) == max(new_ser)
    for file in enc.file_paths:
        os.remove(file)