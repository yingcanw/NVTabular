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


def create_tensors(
    gdf, preproc=None, cat_names=None, cont_names=None, label_name=None, to_cpu=False
):
    # insert preprocessor
    # transform cats here
    gdf = gdf[0]
    if preproc:
        preproc.apply_ops(gdf)
    gdf_cats, gdf_conts, gdf_label = gdf[cat_names], gdf[cont_names], gdf[label_name]
    del gdf
    cats, conts, label = {}, {}, {}
    if len(gdf_cats) > 0:
        to_tensor(gdf_cats, torch.long, cats, to_cpu=to_cpu)
    if len(gdf_conts) > 0:
        to_tensor(gdf_conts, torch.float32, conts, to_cpu=to_cpu)
    if len(gdf_label) > 0:
        to_tensor(gdf_label, torch.float32, label, to_cpu=to_cpu)
    del gdf_cats, gdf_label, gdf_conts
    tar_col = cats.keys()
    cats_list = [cats[x] for x in sorted(cats.keys())] if cats else None
    conts_list = [conts[x] for x in sorted(conts.keys())] if conts else None
    label_list = [label[x] for x in sorted(label.keys())] if label else None
    del cats, conts, label
    cats = torch.stack(cats_list, dim=1) if len(cats_list) > 0 else None
    conts = torch.stack(conts_list, dim=1) if len(conts_list) > 0 else None
    label = torch.cat(label_list, dim=0) if len(label_list) > 0 else None
    return cats, conts, label


def to_tensor(gdf: cudf.DataFrame, dtype, tensor_list, to_cpu=False):
    if gdf.empty:
        return
    for column in gdf.columns:
        gdf_col = gdf[column]
        g = gdf_col.to_dlpack()
        t = from_dlpack(g).type(dtype)
        t = t.to(torch.device("cpu")) if to_cpu else t
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
