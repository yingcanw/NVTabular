import ds_itr.dl_encoder as encoder
import ds_itr.ds_iterator as ds
import cudf
import pytest
import torch
from tests.fixtures import *


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_dl_encoder_fit_transform_fim(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    # create file iterator to go through the
    enc = encoder.DLLabelEncoder("name-string")
    enc.fit(df_expect["name-string"])
    new_ser = enc.transform(df_expect["name-string"])
    unis = set(df_expect["name-string"])
    assert len(unis) == max(new_ser)


@pytest.mark.parametrize("batch", [0])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_dl_encoder_fit_transform_ltm(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    data_itr = ds.GPUDatasetIterator(
        paths[0], batch_size=batch, gpu_memory_frac=0.0001, names=names
    )
    enc = encoder.DLLabelEncoder("name-string", path=str(datasets['category']), limit_frac=0.000001)
    for chunk in data_itr:
        enc.fit(chunk["name-string"])
    new_ser = enc.transform(df_expect["name-string"])
    unis = set(df_expect["name-string"])
    assert len(unis) == max(new_ser)
