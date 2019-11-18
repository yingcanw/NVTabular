import torch
from torch import _utils
from fastai.torch_core import to_device
import cudf
from torch.utils.dlpack import from_dlpack
from ds_itr.ds_iterator import GPUFileIterator


class FileItrDataset(torch.utils.data.IterableDataset):
    gpu_itr = None

    def __init__(self, file, **kwargs):
        self.gpu_itr = GPUFileIterator(file, **kwargs)

    def __iter__(self):
        return self.gpu_itr.__iter__()

    def __len__(self):
        return len(self.gpu_itr)


def create_tensors(gdf, preproc=None, cat_names=None, cont_names=None, label_name=None):
    # insert preprocessor
    # transform cats here
    gdf = gdf[0]
    if preproc:
        preproc.apply_ops(gdf)
    gdf_cats, gdf_conts, gdf_label = gdf[cat_names], gdf[cont_names], gdf[label_name]
    del gdf
    cats, conts, label = {}, {}, {}
    if len(gdf_cats) > 0:
        to_tensor(gdf_cats, torch.long, cats)
    if len(gdf_conts) > 0:
        to_tensor(gdf_conts, torch.float32, conts)
    if len(gdf_label) > 0:
        to_tensor(gdf_label, torch.float32, label, non_target=False)
    del gdf_cats, gdf_label, gdf_conts
    tar_col = cats.keys()
    cats_list = [cats[x] for x in sorted(cats.keys())] if cats else None
    conts_list = [conts[x] for x in sorted(conts.keys())] if conts else None
    label_list = [label[x] for x in sorted(label.keys())] if label else None
    del cats, conts, label
    cats = torch.stack(cats_list, dim=1) if cats_list else None
    conts = torch.stack(conts_list, dim=1) if conts_list else None
    label = torch.cat(label_list, dim=0) if label_list else None
    return cats, conts, label


def to_tensor(gdf: cudf.DataFrame, dtype, tensor_list, non_target=True):
    if gdf.empty:
        return
    for column in gdf.columns:
        gdf_col = gdf[column]
        g = gdf_col.to_dlpack()
        t = from_dlpack(g).type(dtype)
        if non_target:
            t = t.unsqueeze(1) if gdf.shape[1] == 1 else t
        #             t = t.to(self.to_cpu) if self.to_cpu else t
        tensor_list[column] = (
            t if column not in tensor_list else torch.cat([tensor_list[column], t])
        )
        del g


class DLCollator:
    transform = None
    preproc = None
    cat_names = []
    cont_names = []
    label_name = []

    def __init__(
        self,
        transform=create_tensors,
        preproc=None,
        cat_names=None,
        cont_names=None,
        label_name=None,
    ):
        self.transform = transform
        self.preproc = preproc
        if self.preproc:
            self.cat_names = self.preproc.cat_names
            self.cont_names = self.preproc.cont_names
            self.label_name = self.preproc.label_name
        else:
            self.cat_names = cat_names
            self.cont_names = cont_names
            self.label_name = label_name

    def gdf_col(self, gdf):
        batch = self.transform(
            gdf,
            preproc=self.preproc,
            cat_names=self.cat_names,
            cont_names=self.cont_names,
            label_name=self.label_name,
        )
        return (batch[0], batch[1]), batch[2].long()


class DLDataLoader(torch.utils.data.DataLoader):
    def __len__(self):
        return len(self.dataset)
