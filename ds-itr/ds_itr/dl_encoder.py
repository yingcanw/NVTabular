import numpy as np
import cudf
import rmm
import numba
import ds_itr.ds_iterator as ds_itr
import ds_itr.ds_writer as ds_wtr
import os


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
    def __init__(self, col, *args, **kwargs):
        self._cats = kwargs["cats"] if "cats" in kwargs else cudf.Series([None])
        # writer needs to be mapped to same file in folder.
        self.folder_path = kwargs.get("path", col)
        self.col = col
        self.limit_frac = kwargs.get("limit_frac", 0.1)

    def transform(self, y: cudf.Series, unk_idx=0) -> cudf.Series:
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
        y = _enforce_str(y)
        encoded = None
        if os.path.exists(self.col):
            # some cats in memory some in disk
            file_paths = [
                f"{self.col}/{x}" for x in os.listdir(self.col) if x.endswith("parquet")
            ]
            if file_paths:
                chunks = ds_itr.GPUDatasetIterator(file_paths)
                encoded = cudf.Series()
                rec_count = 0
                # chunks represents a UNIQUE set of categorical representations
                for chunk in chunks:
                    # must reconstruct encoded series from multiple parts
                    part_encoded = cudf.Series(
                        y.label_encoding(chunk[self.col].values_to_string())
                    )
                    # zero out unknowns
                    part_encoded.replace(-1, 0)
                    part_encoded[part_encoded > 0].add(rec_count)
                    # continually add chunks to encoded to get full batch
                    encoded = (
                        part_encoded if encoded.empty else encoded.add(part_encoded)
                    )
                    rec_count = rec_count + len(chunk)
        else:
            # all cats in memory
            encoded = cudf.Series(y.label_encoding(self._cats.values_to_string()))
        encoded = encoded.fillna(0)
        # slicing to fix https://github.com/rapidsai/cudf/issues/3785
        return encoded[:].replace(-1, 0)

    def series_size(self, s):
        if hasattr(s, "str"):
            return s.str.device_memory()
        else:
            return s.dtype.itemsize * len(s)

    def fit(self, y: cudf.Series):
        y = _enforce_str(y)
        if self._cats.empty:
            self._cats = self.one_cycle(y).unique()
            return
        self._cats = self._cats.append(self.one_cycle(y)).unique()
        # check if enough space to leave in gpu memory if category doubles in size
        if self.series_size(self._cats) > (
            numba.cuda.current_context().get_memory_info()[0] * self.limit_frac
        ):
            # first time dumping into file
            if not os.path.exists(self.col):
                os.makedirs(self.col)
            self.dump_cats()

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
        x.to_parquet(self.col)
        self._cats = cudf.Series()

    def one_cycle(self, compr):
        # compr is already a list of unique values to check against
        if os.path.exists(self.col):
            file_paths = [
                f"{self.col}/{x}" for x in os.listdir(self.col) if x.endswith("parquet")
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
        return str(self._cats.values_to_string())
