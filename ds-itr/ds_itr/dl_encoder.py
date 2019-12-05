import numpy as np
import nvcategory
import warnings
import cudf
import rmm


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
    def __init__(self, *args, **kwargs):
        self._cats: nvcategory.nvcategory = None
        self._dtype = None
        self._fitted: bool = False

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model must first be .fit()")

    def fit(self, y: cudf.Series) -> "LabelEncoder":
        """
        Fit a LabelEncoder (nvcategory) instance to a set of categories
        Parameters
        ---------
        y : cudf.Series
            Series containing the categories to be encoded. It's elements
            may or may not be unique
        Returns
        -------
        self : LabelEncoder
            A fitted instance of itself to allow method chaining
        """
        self._dtype = y.dtype

        y = _enforce_str(y)

        self._cats = nvcategory.from_strings(y.data)
        self._fitted = True
        return self

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
        self._check_is_fitted()
        y = _enforce_str(y)
        encoded = cudf.Series(
            nvcategory.from_strings(y.data).set_keys(self._cats.keys()).values()
        )

        if encoded.isin([-1]).any() and unk_idx < 0:
            raise KeyError("Attempted to encode unseen key")
        return encoded.replace(-1, unk_idx)

    def fit_transform(self, y: cudf.Series) -> cudf.Series:
        """
        Simultaneously fit and transform an input
        This is functionally equivalent to (but faster than)
        `LabelEncoder().fit(y).transform(y)`
        """
        self._dtype = y.dtype

        # Convert y to nvstrings series, if it isn't one
        y = _enforce_str(y)

        # Bottleneck is here, despite everything being done on the device
        self._cats = nvcategory.from_strings(y.data)

        self._fitted = True
        arr: rmm.device_array = rmm.device_array(y.data.size(), dtype=np.int32)
        self._cats.values(devptr=arr.device_ctypes_pointer.value)
        return cudf.Series(arr)

    def inverse_transform(self, y: cudf.Series) -> cudf.Series:
        """ Revert ordinal label to original label
        Parameters
        ----------
        y : cudf.Series, dtype=int32
            Ordinal labels to be reverted
        Returns
        -------
        reverted : cudf.Series
            Reverted labels
        """
        # check LabelEncoder is fitted
        self._check_is_fitted()
        # check input type is cudf.Series
        if not isinstance(y, cudf.Series):
            raise TypeError("Input of type {} is not cudf.Series".format(type(y)))

        # check if y's dtype is np.int32, otherwise convert it
        y = _enforce_npint32(y)

        # check if ord_label out of bound
        ord_label = y.unique()
        category_num = len(self._cats.keys())
        for ordi in ord_label:
            if ordi < 0 or ordi >= category_num:
                raise ValueError("y contains previously unseen label {}".format(ordi))
        # convert ordinal label to string label
        reverted = cudf.Series(
            self._cats.gather_strings(y.data.mem.device_ctypes_pointer.value, len(y))
        )

        return reverted

    def update_fit(self, y: cudf.Series):
        y = _enforce_str(y)
        self._cats = self._cats.add_strings(y.unique().data)
        self._cats = nvcategory.from_strings(self._cats.keys())
