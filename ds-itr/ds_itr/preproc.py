import yaml
import warnings

import numpy as np
import cudf
from ds_itr.dl_encoder import DLLabelEncoder
from ds_itr.ds_writer import DatasetWriter

try:
    import cupy as cp
except ImportError:
    import numpy as cp

warnings.filterwarnings("ignore")


def _shuffle_part(gdf):
    sort_key = "__sort_index__"
    arr = cp.arange(len(gdf))
    cp.random.shuffle(arr)
    gdf[sort_key] = cudf.Series(arr)
    return gdf.sort_values(sort_key).drop(columns=[sort_key])


class TabularProc:
    def __init__(self, **kwargs):
        self.means = {}
        self.stds = {}
        self.counts = {}
        self.medians = {}
        self.encoders = {}

    def set_stats(self, stats):
        for key, stat in stats.items():
            setattr(self, key, stat)

    def apply_op(self, gdf, cat_names: list, cont_names: list):
        raise NotImplementedError

    @property
    def req_stats(self):
        raise NotImplementedError


class Normalize(TabularProc):
    """ Normalize the continuous variables.
    """

    @property
    def req_stats(self):
        return ["moments"]

    def apply_op(self, gdf, cat_names: list, cont_names: list):
        if not cont_names or not self.stds:
            return gdf
        return self.apply_mean_std(gdf, cont_names)

    def apply_mean_std(self, gdf, cont_names):
        for name in cont_names:
            if self.stds[name] > 0:
                gdf[name] = (gdf[name] - self.means[name]) / (self.stds[name])
            gdf[name] = gdf[name].astype("float32")
        return gdf


class FillMissing(TabularProc):
    MEDIAN = "median"
    CONSTANT = "constant"

    def __init__(self, fill_strategy=MEDIAN, fill_val=0, add_col=False):
        self.fill_strategy = fill_strategy
        self.fill_val = fill_val
        self.add_col = add_col
        self.filler = {}

    @property
    def req_stats(self):
        return ["medians"]

    def initialize(self, itr, cat_names: list, cont_names: list):
        if not cont_names:
            return
        self.get_filler(itr, cat_names, cont_names)

    def apply_op(self, gdf, cat_names: list, cont_names: list):
        if not cont_names or not self.filler:
            return gdf
        return self.apply_filler(gdf, cat_names, cont_names)

    def get_filler(self, itr, cat_names, cont_names):
        cur_filler = {}
        if self.fill_strategy == self.MEDIAN:
            cur_filler = self.medians
        else:
            raise NotImplementedError
        self.filler.update(cur_filler)

    def apply_filler(self, gdf, cat_names, cont_names):
        na_names = [name for name in cont_names if gdf[name].isna().sum()]
        if self.add_col:
            gdf = self.add_na_indicators(gdf, na_names, cont_names)
        for col in na_names:
            gdf[col].fillna(np.float32(self.filler[col]), inplace=True)
        return gdf

    def add_na_indicators(self, gdf: cudf.DataFrame, na_names, cat_names):
        for name in na_names:
            name_na = name + "_na"
            gdf[name_na] = gdf[name].isna()
            if name_na not in cat_names:
                cat_names.append(name_na)
        return gdf


class Categorify(TabularProc):
    """ Transform the categorical variables to that type.
    """

    embed_sz = {}
    cat_names = []

    @property
    def req_stats(self):
        return ["encoders"]

    def apply_op(self, gdf, cat_names: list, cont_names: list):
        if not cat_names:
            return gdf
        self.cat_names.extend(cat_names)
        self.cat_names = list(set(self.cat_names))
        cat_names = [name for name in cat_names if name in gdf.columns]
        for name in cat_names:
            gdf[name] = self.encoders[name].transform(gdf[name])
            gdf[name] = gdf[name].astype("int64")
        return gdf

    def get_emb_sz(self, encoders, cat_names):
        work_in = {}
        for key, val in encoders.items():
            work_in[key] = len(val._cats.keys()) + 1
        ret_list = [(n, self.def_emb_sz(work_in, n)) for n in sorted(cat_names)]
        return ret_list

    def emb_sz_rule(self, n_cat: int) -> int:
        return min(16, round(1.6 * n_cat ** 0.56))

    def def_emb_sz(self, classes, n, sz_dict=None):
        """Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`.
        """
        sz_dict = sz_dict if sz_dict else {}
        n_cat = classes[n]
        sz = sz_dict.get(n, int(self.emb_sz_rule(n_cat)))  # rule of thumb
        self.embed_sz[n] = sz
        return n_cat, sz


class Preprocessor:
    def __init__(
        self, cat_names=None, cont_names=None, label_name=None, ops=None, to_cpu=True
    ):
        self.cat_names = cat_names or []
        self.cont_names = cont_names or []
        self.label_name = label_name or []
        self.ops = ops or []
        self.to_cpu = to_cpu
        self.clear_stats()

    def write_to_dataset(
        self, path, itr, apply_ops=False, nfiles=1, shuffle=True, **kwargs
    ):
        """ Write data to shuffled parquet dataset.
        """
        writer = DatasetWriter(path, nfiles=nfiles)

        for gdf in itr:
            if apply_ops:
                gdf = self.apply_ops(gdf)
            writer.write(gdf, shuffle=shuffle)
        writer.write_metadata()
        return

    def pq_to_pq_processed(
        self,
        indir,
        outdir,
        columns=None,
        shuffle=True,
        apply_ops=True,
        chunk_size=None,
        **kwargs
    ):
        """ Read parquet files and write to new dataset
        """

        # TODO: WARNING -- This method is still a work in progress!!
        # NOTE: There will be memory problems if the files are large
        #       compared to GPU memory.  Need to add check here.

        import dask_cudf

        # Read dataset - Each dask task will read an entire file
        gddf = dask_cudf.read_parquet(
            indir,
            index=False,
            columns=columns,
            split_row_groups=False,
            gather_statistics=True,
        )

        # Shuffle the file (if desired)
        if shuffle:
            gddf = gddf.map_partitions(_shuffle_part)

        # Apply Operations (if desired)
        if apply_ops:
            gddf = gddf.map_partitions(self.apply_ops)

        # Write each partition to an output parquet file
        # (row groups correspond to `chunk_size`)
        gddf.to_parquet(
            outdir, write_index=False, chunk_size=chunk_size, engine="pyarrow"
        )

    def update_stats(self, itr):
        """ Gather necessary column statistics in single pass.
        """

        stats = []
        for op in self.ops:
            if hasattr(op, "req_stats"):
                for stat in op.req_stats:
                    if not stat in stats:
                        stats.append(stat)

        def _apply_stat_func(gdf, type="itr"):
            for stat in stats:
                attr = "get_" + stat + "_" + type
                getattr(self, attr)(gdf, self.cont_names, self.cat_names)

        if stats:
            for gdf in itr:
                _apply_stat_func(gdf, type="itr")
            _apply_stat_func(None, type="post")

        self.set_op_stats()

    def set_op_stats(self):
        for op in self.ops:
            op.set_stats(
                {
                    "medians": self.medians,
                    "means": self.means,
                    "stds": self.stds,
                    "counts": self.counts,
                    "encoders": self.encoders,
                }
            )

    def save_stats(self, path):

        host_categories = {}
        for col in self.categories:
            host_categories[col] = self.categories[col].to_host()

        data = {
            "batch_medians": self.batch_medians,
            "medians": self.medians,
            "means": self.means,
            "vars": self.vars,
            "stds": self.stds,
            "counts": self.counts,
            "host_categories": host_categories,
        }

        with open(path, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_stats(self, path):
        def _set_stats(self, stats_dict):
            for key, stat in stats_dict.items():
                if key == "host_categories":
                    self.encoders_from_host_cats(stat)
                else:
                    setattr(self, key, stat)

        if isinstance(path, dict):
            _set_stats(self, path)
        else:
            with open(path, "r") as infile:
                _set_stats(self, yaml.load(infile))
        self.set_op_stats()

    def get_medians_itr(self, gdf, cont_names, cat_names):
        """ Iteration-level median algorithm.
        """

        # TODO: Use more-accurate approach.
        gdf = gdf[cont_names]
        for name in cont_names:
            if name not in self.batch_medians:
                self.batch_medians[name] = []
            col = gdf[name].copy()
            col = col.dropna().reset_index(drop=True).sort_values()
            if len(col) > 1:
                self.batch_medians[name].append(float(col[len(col) // 2]))
            else:
                self.batch_medians[name].append(0.0)
        return

    def get_medians_post(self, *args):
        """ Finalize median algorithm.
        """
        for col, val in self.batch_medians.items():
            self.batch_medians[col].sort()
            self.medians[col] = float(
                self.batch_medians[col][len(self.batch_medians[col]) // 2]
            )
        return

    def get_moments_itr(self, gdf, cont_names, cat_names):
        """ Iteration-level moment algorithm (mean/std).
        """
        gdf_cont = gdf[cont_names]
        for col in cont_names:
            if col not in self.counts:
                self.counts[col] = 0.0
                self.means[col] = 0.0
                self.vars[col] = 0.0
                self.stds[col] = 0.0

            # TODO: Harden this routine to handle 0-division.
            #       This algo may also break/overflow at scale.

            n1 = self.counts[col]
            n2 = float(len(gdf_cont))

            v1 = self.vars[col]
            v2 = gdf_cont[col].var()

            m1 = self.means[col]
            m2 = gdf_cont[col].mean()

            self.counts[col] += n2
            self.means[col] = (m1 * n1 + m2 * n2) / self.counts[col]

            #  Variance
            t1 = n1 * v1
            t2 = n2 * v2
            t3 = n1 * ((m1 - self.means[col]) ** 2)
            t4 = n2 * ((m2 - self.means[col]) ** 2)
            t5 = n1 + n2
            self.vars[col] = (t1 + t2 + t3 + t4) / t5
        return

    def get_moments_post(self, *args):
        """ Finalize statistical-moments algoprithm.
        """
        for col in self.vars.keys():
            self.stds[col] = float(np.sqrt(self.vars[col]))
        return

    def get_encoders_itr(self, gdf, cont_names, cat_names):
        """ Iteration-level categorical encoder update.
        """
        if not cat_names:
            return
        for name in cat_names:
            if not name in self.encoders:
                self.encoders[name] = DLLabelEncoder()
                self.encoders[name].fit(gdf[name])
            else:
                self.encoders[name].update_fit(gdf[name])
        return

    def get_encoders_post(self, *args):
        """ Finalize categorical encoders (get categories).
        """
        for name, val in self.encoders.items():
            self.categories[name] = val._cats.keys()
        return

    def encoders_from_host_cats(self, host_categories):
        """ Update encoders/categories using host_categories.
        """
        for name, cats in host_categories.items():
            self.encoders[name] = DLLabelEncoder()
            self.encoders[name].fit(cudf.Series(cats))
            self.categories[name] = self.encoders[name]._cats.keys()
        return

    def apply_ops(self, gdf):
        for op in self.ops:
            gdf = op.apply_op(gdf, self.cat_names, self.cont_names)
        return gdf

    def clear_stats(self):

        # Statistics
        self.batch_medians = {}
        self.medians = {}
        self.means = {}
        self.vars = {}
        self.stds = {}
        self.counts = {}
        self.encoders = {}
        self.categories = {}
        self.set_op_stats()

        self.cats, self.conts, self.label = {}, {}, {}

    def ds_to_tensors(self, itr, apply_ops=True):
        import torch
        from torch.utils.dlpack import from_dlpack

        def _to_tensor(gdf: cudf.DataFrame, dtype, tensor_list, non_target=True):
            print(gdf.shape)
            if gdf.empty:
                return
            for column in gdf.columns:
                gdf_col = gdf[column]
                g = gdf_col.to_dlpack()
                t = from_dlpack(g).type(dtype)
                if non_target:
                    t = t.unsqueeze(1) if gdf.shape[1] == 1 else t
                t = t.to(torch.device("cpu")) if self.to_cpu else t
                tensor_list[column] = (
                    t
                    if column not in tensor_list
                    else torch.cat([tensor_list[column], t])
                )
                del g

        for gdf in itr:
            if apply_ops:
                gdf = self.apply_ops(gdf)

            gdf_cats, gdf_conts, gdf_label = (
                gdf[self.cat_names],
                gdf[self.cont_names],
                gdf[self.label_name],
            )
            del gdf

            if len(gdf_cats) > 0:
                _to_tensor(gdf_cats, torch.long, self.cats)
            if len(gdf_conts) > 0:
                _to_tensor(gdf_conts, torch.float32, self.conts)
            if len(gdf_label) > 0:
                _to_tensor(gdf_label, torch.float32, self.label, non_target=False)

        cats_list = (
            [self.cats[x] for x in sorted(self.cats.keys())] if self.cats else None
        )
        conts_list = (
            [self.conts[x] for x in sorted(self.conts.keys())] if self.conts else None
        )
        label_list = (
            [self.label[x] for x in sorted(self.label.keys())] if self.label else None
        )

        # Change cats, conts to dim=1 for column dim=0 for df sub section
        cats = torch.stack(cats_list, dim=1) if cats_list else None
        conts = torch.stack(conts_list, dim=1) if conts_list else None
        label = torch.cat(label_list, dim=0) if label_list else None
        return cats, conts, label
