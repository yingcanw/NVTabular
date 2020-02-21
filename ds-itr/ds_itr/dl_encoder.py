import numpy as np
import cudf
import pandas as pd
import rmm
import numba
import ds_itr.ds_iterator as ds_itr
import ds_itr.ds_writer as ds_wtr
import os
import psutil
from cudf.utils import cudautils
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
    def __init__(self, col, cats=None, path=None, limit_frac=0.1, 
                 gpu_mem_util_limit = 0.8, cpu_mem_util_limit = 0.8, 
                 gpu_mem_trans_use = 0.8, file_paths=None):
        # required because cudf.series does not compute bool type
        self._cats_counts = cudf.Series([]) 
        self._cats_counts_host = None
        self._cats = cats if type(cats) == cudf.Series else cudf.Series([cats])
        self.host_mem_used = False
        self.disk_used = False
        # writer needs to be mapped to same file in folder.
        self.path = path or os.path.join(os.getcwd(), 'label_encoders')
        self.folder_path = os.path.join(self.path, col)
        self.file_paths = file_paths or []
        # incase there are files already in the directory, ignored
        self.ignore_files = []
        if os.path.exists(self.folder_path):
            self.ignore_files = [os.path.join(self.folder_path, x) for x in os.listdir(self.folder_path) if x.endswith("parquet") and x not in self.file_paths
                ]
        self.col = col
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.cpu_mem_util_limit = cpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.sub_cats_size = 50000

    def label_encoding(self, vals, cats, dtype=None, na_sentinel=-1):
        if dtype is None:
            dtype = min_scalar_type(len(cats), 32)

        order = cudf.Series(cudautils.arange(len(vals)))
        codes = cats.index

        value = cudf.DataFrame({"value": cats, "code": codes})
        codes = cudf.DataFrame({"value": vals.copy(), "order": order})
        codes = codes.merge(value, on="value", how="left")
        codes = codes.sort_values("order")["code"].fillna(na_sentinel)
        cats.name = None  # because it was mutated above

        return codes._copy_construct(name=None, index=vals.index)

    def transform_old(self, y: cudf.Series, unk_idx=0) -> cudf.Series:
        """
        Transform an input into its categorical keys.
        This is intended for use with small inputs relative to the size of the
        dataset. For fitting and transforming an entire dataset, prefer
        `fit_transform`.
        Parameters
        ----------
        y : cudf.Series
            Input keys to be transformed. Its values should match the
            categories given to `fit`
        Returns
        ------
        encoded : cudf.Series
            The ordinally encoded input series
        Raises
        ------
        KeyError
            if a category appears that was not seen in `fit`
        """
        # Need to watch out for None calls now
        y = _enforce_str(y).reset_index(drop=True)
        encoded = None
        if os.path.exists(self.folder_path) and self.file_paths:
            # some cats in memory some in disk
            file_paths = [os.path.join(self.folder_path, x) for x in os.listdir(self.folder_path) if x.endswith("parquet") and x not in self.file_paths + self.ignore_files
            ]
            self.file_paths.extend(file_paths)
            self.file_paths = list(set(self.file_paths))
            if self.file_paths:
                chunks = ds_itr.GPUDatasetIterator(self.file_paths)
                encoded = cudf.Series()
                rec_count = 0
                # chunks represents a UNIQUE set of categorical representations
                for chunk in chunks:
                    # must reconstruct encoded series from multiple parts
                    # zero out unknowns using na_sentinel
                    part_encoded = cudf.Series(
                        y.label_encoding(chunk[self.col], na_sentinel=0)
                    )
                    # added ref count to all values over zero in series
                    part_encoded = part_encoded + (part_encoded>0).astype("int") * rec_count
                    # continually add chunks to encoded to get full batch
                    encoded = (
                        part_encoded if encoded.empty else encoded.add(part_encoded)
                    )
                    rec_count = rec_count + len(chunk)
        else:
            # all cats in memory
            encoded = cudf.Series(y.label_encoding(self._cats, na_sentinel=0))
        return encoded[:].replace(-1, 0)

    def transform(self, y: cudf.Series, unk_idx=0) -> cudf.Series:
        # Need to watch out for None calls now
        if self.host_mem_used is False and self.disk_used is False:
            encoded = cudf.Series(y.label_encoding(self._cats, na_sentinel=0))
        elif self.disk_used is False:
            avail_gpu_mem = numba.cuda.current_context().get_memory_info()[0]
            self.sub_cats_size = int(avail_gpu_mem * self.gpu_mem_trans_use / self._cats_host.dtype.itemsize)
            i = 0
            encoded = None
            while i < len(self._cats_host):
                sub_cats = cudf.Series(self._cats_host[i:i+self.sub_cats_size])
                if encoded is None:
                    encoded = self.label_encoding(y, sub_cats, na_sentinel=0)
                else:
                    encoded = encoded.add(self.label_encoding(y, sub_cats, na_sentinel=0), fill_value=0)
                i = i + self.sub_cats_size

            sub_cats = cudf.Series([])
        else:
            print("Unload to files")

        return encoded[:].replace(-1, 0)

    def series_size(self, s):
        if hasattr(s, "str"):
            return s.str.device_memory()
        else:
            return s.dtype.itemsize * len(s)

    def fit_old(self, y: cudf.Series):
        y = _enforce_str(y).reset_index(drop=True)
        if self._cats.empty:
            self._cats = self.one_cycle(y)
            return

        self._cats = self._cats.append(self.one_cycle(y)).unique()
        # check if enough space to leave in gpu memory if category doubles in size
        if self.series_size(self._cats) > (
            numba.cuda.current_context().get_memory_info()[0] * self.limit_frac
        ):
            # first time dumping into file
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)

            self.dump_cats()

    def fit(self, y: cudf.Series):
        #y = _enforce_str(y).reset_index(drop=True)
        y_counts = y.value_counts()
        if len(self._cats_counts) == 0:
            self._cats_counts = y_counts
        else:
            self._cats_counts = self._cats_counts.add(y_counts, fill_value=0)

        gpu_mem = numba.cuda.current_context().get_memory_info()
        gpu_mem_util = (gpu_mem[1] - gpu_mem[0]) / gpu_mem[1]
        series_size_gpu = self.series_size(self._cats_counts)

        if series_size_gpu > (gpu_mem[0] * self.limit_frac) or gpu_mem_util > self.gpu_mem_util_limit:
            if self._cats_counts_host is None:
                self._cats_counts_host = self._cats_counts.to_pandas()
            else:
                self._cats_counts_host = self._cats_counts_host.add(self._cats_counts.to_pandas(), fill_value=0)

            self.host_mem_used = True
            self._cats_counts = cudf.Series([]) 

            cpu_mem = psutil.virtual_memory()
            cpu_mem_util = cpu_mem[2]
            series_host_size = self.series_size(self._cats_counts_host)

            if series_host_size > (cpu_mem[1] * self.limit_frac) or cpu_mem_util > self.cpu_mem_util_limit:
                #self.disk_used = True
                print("Unload to files")
        
    # Note: Add 0: None row to _cats.
    def fit_finalize(self, filter_freq=1):
        total_cats = 0
        if self.host_mem_used is False and self.disk_used is False:
            self._cats = cudf.Series(self._cats_counts[self._cats_counts >= filter_freq].index)
            self._cats = cudf.Series([None]).append(self._cats).reset_index(drop=True)
            total_cats = self._cats.shape[0]
        elif self.disk_used is False:
            self._cats_counts_host = self._cats_counts_host.add(self._cats_counts.to_pandas(), fill_value=0)
            self._cats_host = pd.Series(self._cats_counts_host[self._cats_counts_host >= filter_freq].index)
            self._cats_host = pd.Series([None]).append(self._cats_host).reset_index(drop=True)
            self._cats = cudf.Series()
            total_cats = self._cats_host.shape[0]
        else:
            print("Unload to files")

        #self._cats = cudf.Series()
        self._cats_counts = cudf.Series()
        self._cats_counts_host = None

        return total_cats
                
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
        x.to_parquet(self.folder_path)
        self._cats = cudf.Series()
        #should find new file just exported
        new_file_path = [os.path.join(self.folder_path, x) for x in os.listdir(self.folder_path) if x.endswith("parquet") and x not in self.file_paths + self.ignore_files
            ]
        # add file to list
        self.file_paths.extend(new_file_path)
        

    def one_cycle(self, compr):
        # compr is already a list of unique values to check against
        if os.path.exists(self.folder_path):
            file_paths = [
                os.path.join(self.folder_path, x) for x in os.listdir(self.folder_path) if x.endswith("parquet") and x not in self.file_paths + self.ignore_files
            ]
            if file_paths:             
                chunks = ds_itr.GPUDatasetIterator(file_paths)
                for chunk in chunks:
                    compr = self.merge_series(chunk[self.col], compr)
                    if len(compr) == 0:
                        # if nothing is left to compare... bug out
                        break
        return compr

    def __repr__(self):
        return ("{0}(_cats={1!r})".format(type(self).__name__, self._cats.values_to_string()))
