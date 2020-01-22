import yaml
import warnings

import numpy as np
import cudf
from ds_itr.dl_encoder import DLLabelEncoder
from ds_itr.ds_writer import DatasetWriter
from ds_itr.ops import *

try:
    import cupy as cp
except ImportError:
    import numpy as cp


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
        feat_ops=None,
        stat_ops=None,
        df_ops=None,
        to_cpu=True,
        config=None
    ):
        self.reg_funcs = {'FE': self.reg_feat_ops, 'PP':self.reg_df_ops}
        self.cat_names = cat_names or []
        self.cont_names = cont_names or []
        self.label_name = label_name or []
        self.columns_ctx = {}
        self.columns_ctx['all'] = {} 
        self.columns_ctx['continuous'] = {}
        self.columns_ctx['categorical'] = {}
        self.columns_ctx['all']['base'] = cont_names + cat_names
        self.columns_ctx['continuous']['base'] = cont_names
        self.columns_ctx['categorical']['base'] = cat_names
        self.feat_ops = {}
        self.stat_ops = {}
        self.df_ops = {}
        self.stats = {}
        self.task_sets = {}
        self.to_cpu = to_cpu
        if config:
            self.load_config(config)
        else:
            warnings.warn("No Config was loaded, unable to create task list")
        if feat_ops:
            self.reg_feat_ops(feat_ops)
        if stat_ops:
            self.reg_stat_ops(stat_ops)
        if df_ops:
            self.reg_df_ops(df_ops)
        else:
            warnings.warn("No DataFrame Operators were loaded")

        self.clear_stats()

    def reg_feat_ops(self, feat_ops):
        for feat_op in feat_ops:
            self.feat_ops[feat_op._id] = feat_op

    def reg_df_ops(self, df_ops):
        for df_op in df_ops:
            dfop_id, dfop_rs = df_op._id, df_op.req_stats
            self.reg_stat_ops(dfop_rs)
            self.df_ops[dfop_id] = df_op

    def reg_stat_ops(self, stat_ops):
        for stat_op in stat_ops:
            # pull stats, ensure no duplicates
            for stat in stat_op.registered_stats():
                if stat not in self.stats:
                    self.stats[stat] = {}
                else:
                    warnings.warn(
                        f"The following statistic was not added because it already exists: {stat}"
                    )
            # add actual statistic operator, after all stats added
            self.stat_ops[stat_op._id] = stat_op

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
            gddf = gddf.map_partitions(self.apply_ops, meta=self.apply_ops(gddf.head()))

        # Write each partition to an output parquet file
        # (row groups correspond to `chunk_size`)
        gddf.to_parquet(
            outdir, write_index=False, chunk_size=chunk_size, engine="pyarrow"
        )


    def load_config(self, config):
        # separate FE and PP
        self.task_sets = {}
        for task_set in config.keys():
            self.task_sets[task_set] = self.build_tasks(config[task_set])
    
        for t_set, task_list in self.task_sets.items():
            for tup in task_list:
                self.reg_funcs[t_set]([tup[0]])
                
        
    def build_tasks(self, task_dict):
        base_tasks = []
        dep_tasks = []
        for cols, task_list in task_dict.items():
            for task in task_list:
                for op_id, dep_set in task.items():
                    # get op from op_id
                    target_op = all_ops[op_id]
                    if not target_op:
                        warnings.warn(f"""Did not find corresponding op for id: {op_id}. 
                                      If this is a custom operator, check it was properyl
                                      loaded.""")
                        break
                    if hasattr(target_op, 'req_stats'):
                        self.reg_stat_ops(target_op.req_stats)
                    if dep_set:
                        for dep_grp in dep_set:
                            if dep_grp:
                                dep_tasks.append((target_op, cols, dep_grp))
                            else:
                                base_tasks.append((target_op, cols, ['base']))
        return base_tasks + dep_tasks
        
    def update_stats(self, itr):
        """ Gather necessary column statistics in single pass.
        """
        fe_tasks = self.task_sets.get("FE", {})
        pp_tasks = self.task_sets.get("PP", {})
        for gdf in itr:
            #put the FE tasks here roll through them
            for fe_task in fe_tasks:
                op, cols_grp, target_cols = fe_task
                gdf = op.apply_op(gdf, self.columns_ctx, cols_grp, target_cols=target_cols)
            # run the stat ops on the target columns for the parent DF Op in PP tasks list
            for pp_task in pp_tasks:
                op, cols_grp, target_cols = pp_task
                req_ops = op.req_stats
                for op in req_ops:
                    self.stat_ops[op._id].apply_op(gdf, self.columns_ctx, cols_grp, target_cols=target_cols)
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
                    warnings.warn("stat not found,", name)

    def save_stats(self, path):
        stats_drop = {}
        stats_drop["encoders"] = {}
        encoders = self.stats.get("encoders", {})
        for name, enc in encoders.items():
            stats_drop["encoders"][name] = (
                enc.file_paths,
                enc._cats.values_to_string(),
            )
        for name, stat in self.stats.items():
            if name not in stats_drop.keys():
                stats_drop[name] = stat
        with open(path, "w") as outfile:
            yaml.dump(stats_drop, outfile, default_flow_style=False)

    def load_stats(self, path):
        def _set_stats(self, stats_dict):
            for key, stat in stats_dict.items():
                self.stats[key] = stat

        if isinstance(path, dict):
            _set_stats(self, path)
        else:
            with open(path, "r") as infile:
                _set_stats(self, yaml.load(infile))
        encoders = self.stats.get("encoders", {})
        for col, cats in encoders.items():
            self.stats["encoders"][col] = DLLabelEncoder(
                col, file_paths=cats[0], cats=cudf.Series(cats[1])
            )

    def apply_ops(self, gdf):
        # run the PP ops 
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

    def ds_to_tensors(self, itr, apply_ops=True):
        import torch
        from torch.utils.dlpack import from_dlpack

        def _to_tensor(gdf: cudf.DataFrame, dtype, tensor_list, to_cpu=False):
            if gdf.empty:
                return
            for column in gdf.columns:
                gdf_col = gdf[column]
                g = gdf_col.to_dlpack()
                t = from_dlpack(g).type(dtype)
                t = t.to(torch.device("cpu")) if to_cpu else t
                tensor_list[column] = (
                    t
                    if column not in tensor_list
                    else torch.cat([tensor_list[column], t])
                )
                del g

        cats, conts, label = {}, {}, {}
        for gdf in itr:
            if apply_ops:
                for name, feat_op in self.feat_ops.items():
                    gdf = feat_op.apply_op(
                        gdf, self.cont_names, self.cat_names, self.label_name
                    )
                gdf = self.apply_ops(gdf)

            gdf_cats, gdf_conts, gdf_label = (
                gdf[self.cat_names],
                gdf[self.cont_names],
                gdf[self.label_name],
            )
            del gdf

            if len(gdf_cats) > 0:
                _to_tensor(gdf_cats, torch.long, cats, to_cpu=self.to_cpu)
            if len(gdf_conts) > 0:
                _to_tensor(gdf_conts, torch.float32, conts, to_cpu=self.to_cpu)
            if len(gdf_label) > 0:
                _to_tensor(gdf_label, torch.float32, label, to_cpu=self.to_cpu)

        cats_list = [cats[x] for x in sorted(cats.keys())] if cats else None
        conts_list = [conts[x] for x in sorted(conts.keys())] if conts else None
        label_list = [label[x] for x in sorted(label.keys())] if label else None

        # Change cats, conts to dim=1 for column dim=0 for df sub section
        cats = torch.stack(cats_list, dim=1) if len(cats_list) > 0 else None
        conts = torch.stack(conts_list, dim=1) if len(conts_list) > 0 else None
        label = torch.cat(label_list, dim=0) if len(label_list) > 0 else None
        return cats, conts, label
