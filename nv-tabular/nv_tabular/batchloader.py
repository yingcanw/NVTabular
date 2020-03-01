import torch
from torch import _utils
from fastai.torch_core import to_device
import cudf
from torch.utils.dlpack import from_dlpack
from nv_tabular.ds_iterator import GPUFileIterator


class FileItrDataset(torch.utils.data.IterableDataset):
    gpu_itr = None

    def __init__(self, file, **kwargs):
        self.gpu_itr = GPUFileIterator(file, **kwargs)

    def __iter__(self):
        return self.gpu_itr.__iter__()

    def __len__(self):
        return len(self.gpu_itr)


class TensorItrDataset(torch.utils.data.IterableDataset):
    tensor_itr = None

    def __init__(self, tensors, **kwargs):
        self.tensor_itr = TensorItr(tensors, **kwargs)

    def __iter__(self):
        return self.tensor_itr.__iter__()

    def __len__(self):
        return len(self.tensor_itr)


class TensorItr:
    """Batch Dataset wrapping Tensors.  
    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
        batch_size: The size of the batch to return
        
        
    """

    def __init__(self, tensors, batch_size=1, pin_memory=False, shuffle=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.batch_size = batch_size
        self.cur_idx = 0
        self.num_samples = tensors[0].size(0)
        if shuffle:
            self.shuffle()

        if pin_memory:
            for tensor in self.tensors:
                tensor.pin_memory()

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __len__(self):
        if self.num_samples % self.batch_size == 0:
            return self.num_samples // self.batch_size
        else:
            return self.num_samples // self.batch_size + 1

    def __next__(self):
        idx = self.cur_idx * self.batch_size
        self.cur_idx += 1
        # Need to handle odd sized batches if data isn't divisible by batchsize
        if idx < self.num_samples and (idx + self.batch_size <= self.num_samples):
            tens = [tensor[idx : idx + self.batch_size] for tensor in self.tensors]
            return (tens[0], tens[1]), tens[2]
        elif idx < self.num_samples and idx + self.batch_size > self.num_samples:
            tens = [tensor[idx:] for tensor in self.tensors]
            return (tens[0], tens[1]), tens[2]
        else:
            raise StopIteration

    def shuffle(self):
        idx = torch.randperm(self.num_samples, dtype=torch.int64)
        self.tensors = [tensor[idx] for tensor in self.tensors]


        
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
        
def create_tensors(preproc, itr=None, gdf=None, apply_ops=True):
    cats, conts, label = {}, {}, {}
    if itr:
        for gdf in itr:
            process_one_df(gdf, preproc, cats, conts, label, apply_ops=apply_ops)
    elif gdf:
        process_one_df(gdf, preproc, cats, conts, label, apply_ops=apply_ops)

    cats_list = (
        [
            cats[x]
            for x in sorted(cats.keys(), key=lambda entry: entry.split("_")[0])
        ]
        if cats
        else None
    )
    conts_list = [conts[x] for x in sorted(conts.keys())] if conts else None
    label_list = [label[x] for x in sorted(label.keys())] if label else None

    # Change cats, conts to dim=1 for column dim=0 for df sub section
    cats = torch.stack(cats_list, dim=1) if len(cats_list) > 0 else None
    conts = torch.stack(conts_list, dim=1) if len(conts_list) > 0 else None
    label = torch.cat(label_list, dim=0) if len(label_list) > 0 else None
    return cats, conts, label


def get_final_cols(preproc):
    if not 'cols' in preproc.columns_ctx['final']:
        preproc.create_final_cols()
    cat_names = sorted(
        preproc.columns_ctx["final"]["cols"]["categorical"],
        key=lambda entry: entry.split("_")[0],
    )
    cont_names = sorted(preproc.columns_ctx["final"]["cols"]["continuous"])
    label_name = sorted(preproc.columns_ctx["final"]["cols"]["label"])
    return cat_names, cont_names, label_name

def process_one_df(gdf, preproc, cats, conts, label, apply_ops=True):
    if apply_ops:
        gdf = preproc.apply_ops(gdf)
    
    cat_names, cont_names, label_name = get_final_cols(preproc)

    gdf_cats, gdf_conts, gdf_label = (
        gdf[cat_names],
        gdf[cont_names],
        gdf[label_name],
    )
    del gdf

    if len(gdf_cats) > 0:
        _to_tensor(gdf_cats, torch.long, cats, to_cpu=preproc.to_cpu)
    if len(gdf_conts) > 0:
        _to_tensor(gdf_conts, torch.float32, conts, to_cpu=preproc.to_cpu)
    if len(gdf_label) > 0:
        _to_tensor(gdf_label, torch.float32, label, to_cpu=preproc.to_cpu)



class DLCollator:
    transform = None
    preproc = None
    


    def __init__(
        self,
        transform=create_tensors,
        preproc=None
    ):
        self.transform = transform
        self.preproc = preproc


    def gdf_col(self, gdf):
        batch = self.transform(
            self.preproc, gdf=gdf[0]
        )
        return (batch[0], batch[1]), batch[2].long()


class DLDataLoader(torch.utils.data.DataLoader):
    def __len__(self):
        return len(self.dataset)
