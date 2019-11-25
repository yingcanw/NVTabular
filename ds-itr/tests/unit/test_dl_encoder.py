import ds_itr.dl_encoder as encoder
import cudf
import pytest
import torch
from tests.fixtures import *


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_dl_encoder_fit_transform(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    enc = encoder.DLLabelEncoder()
    new_ser = enc.fit_transform(df_expect["name-string"])
    unis = set(new_ser)
    assert len(unis) == enc._cats.keys_size()


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_dl_encoder_fit_transform_inverse(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    enc = encoder.DLLabelEncoder()
    new_ser = enc.fit_transform(df_expect["name-string"])
    unis = set(new_ser)
    assert len(unis) == enc._cats.keys_size()
    inv_df = enc.inverse_transform(new_ser)
    assert inv_df.tolist() == df_expect["name-string"].tolist()
