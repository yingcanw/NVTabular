import numpy as np
import cudf
import pandas as pd
import rmm
import numba
import nv_tabular.ds_iterator as ds_itr
import nv_tabular.ds_writer as ds_wtr
import os
import psutil
import uuid
import cupy as cp

from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_datetime_dtype,
    is_list_like,
    is_scalar,
    min_scalar_type,
    to_cudf_compatible_scalar,
)


def _enforce_str(y: cudf.Series) -> cudf.Series:
    """
    Ensure that nvcategory is being given strings
    """
    if y.dtype != "object":
        return y.astype("str")
    return y


def _enforce_npint32(y: cudf.Series) -> cudf.Series:
    if y.dtype != np.int32:
        return y.astype(np.int32)
    return y


class DLLabelEncoder(object):
    def __init__(
        self,
        col,
        cats=None,
        path=None,
        use_frequency=False,
        freq_threshold=0,
        limit_frac=0.1,
        gpu_mem_util_limit=0.8,
        cpu_mem_util_limit=0.8,
        gpu_mem_trans_use=0.8,
        file_paths=None,
    ):

        # required because cudf.series does not compute bool type
        self._cats_counts = cudf.Series([])
        self._cats_counts_host = None
        self._cats_host = None
        self._cats_parts = []
        self._cats_host = cats.to_pandas() if type(cats) == cudf.Series else cats
        # writer needs to be mapped to same file in folder.
        self.path = path or os.path.join(os.getcwd(), "label_encoders")
        self.folder_path = os.path.join(self.path, col)
        self.file_paths = file_paths or []
        # incase there are files already in the directory, ignored
        self.ignore_files = []
        if os.path.exists(self.folder_path):
            self.ignore_files = [
                os.path.join(self.folder_path, x)
                for x in os.listdir(self.folder_path)
                if x.endswith("parquet") and x not in self.file_paths
            ]
        self.col = col
        self.use_frequency = use_frequency
        self.freq_threshold = freq_threshold
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.cpu_mem_util_limit = cpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.cat_exp_count = 0

    def label_encoding(self, vals, cats, dtype=None, na_sentinel=-1):
        if dtype is None:
            dtype = min_scalar_type(len(cats), 32)

        order = cudf.Series(cp.arange(len(vals)))
        codes = cats.index

        value = cudf.DataFrame({"value": cats, "code": codes})
        codes = cudf.DataFrame({"value": vals.copy(), "order": order})
        codes = codes.merge(value, on="value", how="left")
        codes = codes.sort_values("order")["code"].fillna(na_sentinel)
        cats.name = None  # because it was mutated above
        return codes._copy_construct(name=None, index=vals.index)

    def transform(self, y: cudf.Series, unk_idx=0) -> cudf.Series:
        if self._cats_host is None:
            raise Exception("Encoder was not fit!")

        if len(self._cats_host) == 0:
            raise Exception("Encoder was not fit!")

        avail_gpu_mem = numba.cuda.current_context().get_memory_info()[0]
        sub_cats_size = int(
            avail_gpu_mem * self.gpu_mem_trans_use / self._cats_host.dtype.itemsize
        )
        i = 0
        encoded = None
        while i < len(self._cats_host):
            sub_cats = cudf.Series(self._cats_host[i : i + sub_cats_size])
            if encoded is None:
                encoded = self.label_encoding(y, sub_cats, na_sentinel=0)
            else:
                encoded = encoded.add(
                    self.label_encoding(y, sub_cats, na_sentinel=0), fill_value=0
                )
            i = i + sub_cats_size

        sub_cats = cudf.Series([])
        return encoded[:].replace(-1, 0)

    def series_size(self, s):
        if hasattr(s, "str"):
            return s.str.device_memory()
        else:
            return s.dtype.itemsize * len(s)

    # Returns GPU available space and utilization
    def get_gpu_mem_info(self):
        gpu_mem = numba.cuda.current_context().get_memory_info()
        gpu_mem_util = (gpu_mem[1] - gpu_mem[0]) / gpu_mem[1]
        return gpu_mem[0], gpu_mem_util

    # Returns CPU available space and utilization
    def get_cpu_mem_info(self):
        cpu_mem = psutil.virtual_memory()
        cpu_mem_util = (cpu_mem[0] - cpu_mem[1]) / cpu_mem[0]
        return cpu_mem[1], cpu_mem_util

    def fit(self, y: cudf.Series):
        if self.use_frequency:
            self.fit_freq(y)
        else:
            self.fit_unique(y)

    def fit_finalize(self):
        if self.use_frequency:
            return self.fit_freq_finalize()
        else:
            return self.fit_unique_finalize()

    def fit_unique(self, y: cudf.Series):
        y_uniqs = y.unique()
        self._cats_parts.append(y_uniqs.to_pandas())

    def fit_unique_finalize(self):
        y_uniqs = (
            cudf.Series([])
            if self._cats_host is None
            else cudf.from_pandas(self._cats_host)
        )
        cats_uniqs_host = []
        for i in range(len(self._cats_parts)):
            y_uniqs_part = cudf.from_pandas(self._cats_parts.pop())
            if y_uniqs.shape[0] == 0:
                y_uniqs = y_uniqs_part
            else:
                y_uniqs = y_uniqs.append(
                    y_uniqs_part
                ).unique()  # Check merge option as well

        cats = cudf.Series([None]).append(y_uniqs)
        cats = cats.unique().reset_index(drop=True)
        self._cats_host = cats.to_pandas()
        return self._cats_host.shape[0]

    def fit_freq(self, y: cudf.Series):
        y_counts = y.value_counts()
        self._cats_parts.append(y_counts.to_pandas())

    def fit_freq_finalize(self):
        y_counts = cudf.Series([])
        cats_counts_host = []
        for i in range(len(self._cats_parts)):
            y_counts_part = cudf.from_pandas(self._cats_parts.pop())
            if y_counts.shape[0] == 0:
                y_counts = y_counts_part
            else:
                y_counts = y_counts.add(y_counts_part, fill_value=0)
            series_size_gpu = self.series_size(y_counts)

            avail_gpu_mem, gpu_mem_util = self.get_gpu_mem_info()
            if (
                series_size_gpu > (avail_gpu_mem * self.limit_frac)
                or gpu_mem_util > self.gpu_mem_util_limit
            ):
                cats_counts_host.append(y_counts.to_pandas())
                y_counts = cudf.Series([])

        if len(cats_counts_host) == 0:
            cats = cudf.Series(y_counts[y_counts >= self.freq_threshold].index)
            cats = cudf.Series([None]).append(cats).reset_index(drop=True)
            self._cats_host = cats.to_pandas()
        else:
            y_counts_host = cats_counts_host.pop()
            for i in range(len(cats_counts_host)):
                y_counts_host_temp = cats_counts_host.pop()
                y_counts_host = y_counts_host.add(y_counts_host_temp, fill_value=0)

            self._cats_host = pd.Series(
                y_counts_host[y_counts_host >= self.freq_threshold].index
            )
            self._cats_host = (
                pd.Series([None]).append(self._cats_host).reset_index(drop=True)
            )

        return self._cats_host.shape[0]

    def merge_series(self, compr_a, compr_b):
        df, dg = cudf.DataFrame(), cudf.DataFrame()
        df["l1"] = compr_a.nans_to_nulls().dropna()
        dg["l2"] = compr_b.nans_to_nulls().dropna()
        mask = dg["l2"].isin(df["l1"])
        unis = dg.loc[~mask]["l2"].unique()
        return unis

    def dump_cats(self):
        x = cudf.DataFrame()
        x[self.col] = self._cats.unique()
        self.cat_exp_count = self.cat_exp_count + x.shape[0]
        file_id = str(uuid.uuid4().hex) + ".parquet"
        tar_file = os.path.join(self.folder_path, file_id)
        x.to_parquet(tar_file)
        self._cats = cudf.Series()
        # should find new file just exported
        new_file_path = [
            os.path.join(self.folder_path, x)
            for x in os.listdir(self.folder_path)
            if x.endswith("parquet")
            and x not in self.file_paths
            and x not in self.ignore_files
        ]
        # add file to list
        self.file_paths.extend(new_file_path)
        self.file_paths = list(set(self.file_paths))

    def one_cycle(self, compr):
        # compr is already a list of unique values to check against
        if os.path.exists(self.folder_path):
            file_paths = [
                os.path.join(self.folder_path, x)
                for x in os.listdir(self.folder_path)
                if x.endswith("parquet")
                and x not in self.file_paths + self.ignore_files
            ]
            if file_paths:
                chunks = ds_itr.GPUDatasetIterator(file_paths)
                for chunk in chunks:
                    compr = self.merge_series(chunk[self.col], compr)
                    if len(compr) == 0:
                        # if nothing is left to compare... bug out
                        break
        return compr

    def get_cats(self):
        return cudf.from_pandas(self._cats_host).reset_index(drop=True)

    def __repr__(self):
        return "{0}(_cats={1!r})".format(
            type(self).__name__, self.get_cats().values_to_string()
        )
