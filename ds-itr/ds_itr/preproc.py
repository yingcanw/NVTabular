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


class Preprocessor:
    def __init__(
        self,
        cat_names=None,
        cont_names=None,
        label_name=None,
        stat_ops=None,
        df_ops=None,
        to_cpu=True,
    ):
        self.cat_names = cat_names or []
        self.cont_names = cont_names or []
        self.label_name = label_name or []
        self.stat_ops = {}
        self.df_ops = {}
        self.stats = {}
        self.to_cpu = to_cpu
        if stat_ops:
            for stat_op in stat_ops:
                # pull stats, ensure no duplicates
                for stat in stat_op.registered_stats():
                    if stat not in self.stats:
                        self.stats[stat] = {}
                    else:
                        print(
                            f"The following statistic was not added because it already exists: {stat}"
                        )
                        break
                # add actual statistic operator, after all stats added
                self.stat_ops[stat_op._id] = stat_op
        else:
            print("No Statistical Operators were loaded")
        # after stats are loaded, add df_ops with available stats only
        if df_ops:
            for df_op in df_ops:
                dfop_id, dfop_rs = df_op._id, df_op.req_stats
                if all(name in self.stats for name in dfop_rs):
                    self.df_ops[dfop_id] = df_op
                else:
                    print(
                        f"The following df_op was not added because necessary stats not loaded: {dfop_id, dfop_rs}"
                    )
        else:
            print("No DataFrame Operators were loaded")

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
        **kwargs,
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

        #         stats = []
        #         for op in self.ops:
        #             if hasattr(op, "req_stats"):
        #                 for stat in op.req_stats:
        #                     if not stat in stats:
        #                         stats.append(stat)

        for gdf in itr:
            for name, stat_op in self.stat_ops.items():
                stat_op.read_itr(gdf, self.cont_names, self.cat_names, self.label_name)
        for name, stat_op in self.stat_ops.items():
            stat_op.read_fin()
            # missing bubble up to prerprocessor
        self.get_stats()

    def get_stats(self):
        for name, stat_op in self.stat_ops.items():
            stat_vals = stat_op.stats_collected()
            for name, stat in stat_vals:
                if name in self.stats:
                    self.stats[name] = stat
                else:
                    print("stat not found,", name)

    def save_stats(self, path):

        host_categories = {}
        for col, cat in self.stats["categories"].items():
            host_categories[col] = cat.to_host()

        self.stats["host_categories"] = host_categories

        with open(path, "w") as outfile:
            yaml.dump(self.stats, outfile, default_flow_style=False)

    def load_stats(self, path):
        def _set_stats(self, stats_dict):
            for key, stat in stats_dict.items():
                if key == "host_categories":
                    self.encoders_from_host_cats(stat)
                else:
                    self.stats[key] = stat

        if isinstance(path, dict):
            _set_stats(self, path)
        else:
            with open(path, "r") as infile:
                _set_stats(self, yaml.load(infile))

    def apply_ops(self, gdf):
        for name, op in self.df_ops.items():
            gdf = op.apply_op(
                gdf, self.stats, self.cont_names, self.cat_names, self.label_name
            )
        return gdf

    def clear_stats(self):

        for stat, vals in self.stats.items():
            self.stats[stat] = {}

        for statop_id, stat_op in self.stat_ops.items():
            stat_op.clear()

    def encoders_from_host_cats(self, host_categories):
        """ Update encoders/categories using host_categories.
        """
        for name, cats in host_categories.items():
            self.stats["encoders"][name] = DLLabelEncoder()
            self.stats["encoders"][name].fit(cudf.Series(cats))
            self.stats["categories"][name] = self.stats["encoders"][name]._cats.keys()
        return

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
                #                 if non_target:
                #                     t = t.unsqueeze(1) if gdf.shape[1] == 1 else t
                t = t.to(torch.device("cpu")) if self.to_cpu else t
                tensor_list[column] = (
                    t
                    if column not in tensor_list
                    else torch.cat([tensor_list[column], t])
                )
                del g

        cats, conts, label = {}, {}, {}
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
                _to_tensor(gdf_cats, torch.long, cats)
            if len(gdf_conts) > 0:
                _to_tensor(gdf_conts, torch.float32, conts)
            if len(gdf_label) > 0:
                _to_tensor(gdf_label, torch.float32, label)

        cats_list = [cats[x] for x in sorted(cats.keys())] if cats else None
        conts_list = [conts[x] for x in sorted(conts.keys())] if conts else None
        label_list = [label[x] for x in sorted(label.keys())] if label else None

        # Change cats, conts to dim=1 for column dim=0 for df sub section
        cats = torch.stack(cats_list, dim=1) if len(cats_list) > 0 else None
        conts = torch.stack(conts_list, dim=1) if len(conts_list) > 0 else None
        label = torch.cat(label_list, dim=0) if len(label_list) > 0 else None
        return cats, conts, label
