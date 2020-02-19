import cudf
import sys
import numba

#
# Helper Function definitions
#


def _allowable_batch_size(gpu_memory_frac, row_size):
    free_mem, _ = numba.cuda.current_context().get_memory_info()
    gpu_memory = free_mem * gpu_memory_frac
    return max(int(gpu_memory / row_size), 1)


def _get_read_engine(engine, file_path, **kwargs):
    if engine is None:
        engine = file_path.split(".")[-1]
    if not isinstance(engine, str):
        raise TypeError("Expecting engine as string type.")

    if engine == "csv":
        return CSVFileReader(file_path, **kwargs)
    elif engine == "parquet":
        return PQFileReader(file_path, **kwargs)
    else:
        raise ValueError("Unrecognized read engine.")


#
# GPUFileReader Base Class
#


class GPUFileReader:
    def __init__(self, file_path, gpu_memory_frac, batch_size, row_size=None, **kwargs):
        """ GPUFileReader Constructor
        """
        self.file = None
        self.file_path = file_path
        self.row_size = row_size
        self.intialize_reader(gpu_memory_frac, batch_size, **kwargs)

    def intialize_reader(self, **kwargs):
        """ Define necessary file statistics and properties for reader
        """
        raise NotImplementedError()

    def read_file_batch(self, nskip=0, columns=None, **kwargs):
        """ Read a chunk of a tabular-data file

        Parameters
        ----------
        nskip: int
            Row offset
        columns: List[str]
            List of column names to read
        **kwargs:
            Other format-specific key-word arguments

        Returns
        -------
        A CuDF DataFrame
        """
        raise NotImplementedError()

    @property
    def estimated_row_size(self):
        return self.row_size

    def __del__(self):
        """ GPUFileReader Destructor
        """
        if self.file:
            self.file.close()


#
# GPUFileReader Sub Classes (Parquet and CSV Engines)
#


class PQFileReader(GPUFileReader):
    def intialize_reader(self, gpu_memory_frac, batch_size, **kwargs):
        self.reader = cudf.read_parquet
        self.file = open(self.file_path, "rb")

        # Read Parquet-file metadata
        (
            self.num_rows,
            self.num_row_groups,
            self.columns,
        ) = cudf.io.read_parquet_metadata(self.file)
        self.file.seek(0)
        # Use first row-group metadata to estimate memory-rqs
        # NOTE: We could also use parquet metadata here, but
        #       `total_uncompressed_size` for each column is
        #       not representitive of dataframe size for
        #       strings/categoricals (parquet only stores uniques)
        self.row_size = self.row_size or 0
        if self.num_rows > 0 and self.row_size == 0:
            for col in self.reader(self.file, num_rows=1)._columns:
                
                self.row_size += col.dtype.itemsize
            self.file.seek(0)
        # Check if wwe are using row groups
        self.use_row_groups = kwargs.get("use_row_groups", None)
        self.row_group_batch = 1
        self.next_row_group = 0

        # Determine batch size if needed
        if batch_size and not self.use_row_groups:
            self.batch_size = batch_size
            self.use_row_groups = False
        else:
            # Use row size to calculate "allowable" batch size
            gpu_memory_batch = _allowable_batch_size(gpu_memory_frac, self.row_size)
            self.batch_size = min(gpu_memory_batch, self.num_rows)

            # Use row-groups if they meet memory constraints
            rg_size = int(self.num_rows / self.num_row_groups)
            if (self.use_row_groups is None) and (rg_size <= gpu_memory_batch):
                self.use_row_groups = True
            elif self.use_row_groups is None:
                self.use_row_groups = False

            # Determine row-groups per batch
            if self.use_row_groups:
                self.row_group_batch = max(int(gpu_memory_batch / rg_size), 1)

    def read_file_batch(self, nskip=0, columns=None, **kwargs):
#         if self.use_row_groups:
#             row_group_batch = min(
#                 self.row_group_batch, self.num_row_groups - self.next_row_group
#             )
#             chunk = cudf.DataFrame()
#             for i in range(row_group_batch):
#                 add_chunk = self.reader(
#                     self.file_path,
#                     row_group=self.next_row_group,
#                     engine="cudf",
#                     columns=columns,
#                 )
#                 self.next_row_group += 1
#                 chunk = cudf.concat([chunk, add_chunk], axis=0) if chunk else add_chunk
#                 del add_chunk
#             return chunk.reset_index(drop=True)
#         else:
        batch = min(self.batch_size, self.num_rows - nskip)
        return self.reader(
            self.file_path,
            num_rows=batch,
            skip_rows=nskip,
            engine="cudf",
            columns=columns,
        ).reset_index(drop=True)


class CSVFileReader(GPUFileReader):
    def intialize_reader(self, gpu_memory_frac, batch_size, **kwargs):
        self.reader = cudf.read_csv
        self.file = open(self.file_path, "r")

        # Count rows and determine column names
        self.columns = []
        estimate_row_size = False
        if self.row_size is None:
            self.row_size = 0
            estimate_row_size = True

        for i, l in enumerate(self.file):
            pass
        self.file.seek(0)
        self.num_rows = i

        # Use first row to estimate memory-reqs
        names = kwargs.get("names", None)
        dtype = kwargs.get("dtype", None)
        # default csv delim is ","
        sep = kwargs.get("sep", ",")
        self.sep = sep
        self.names = []
        dtype_inf = {}
        snippet = self.reader(
            self.file,
            nrows=min(10, self.num_rows),
            names=names,
            header=False,
            dtype=dtype,
            sep=sep,
        )
        if self.num_rows > 0:
            for i, col in enumerate(snippet.columns):
                if names:
                    name = names[i]
                else:
                    name = col
                self.names.append(name)
            for i, col in enumerate(snippet._columns):
                if estimate_row_size:
                    if col.dtype == "object":
                        # Use maximum of first 10 rows
                        max_size = len(max(col.dropna())) // 2
                        self.row_size += int(max_size)
                    else:
                        self.row_size += col.dtype.itemsize
                dtype_inf[self.names[i]] = col.dtype
        self.dtype = dtype or dtype_inf

        # Determine batch size if needed
        if batch_size:
            self.batch_size = batch_size
        else:
            gpu_memory_batch = _allowable_batch_size(gpu_memory_frac, self.row_size)
            self.batch_size = min(gpu_memory_batch, self.num_rows)

    def read_file_batch(self, nskip=0, columns=None, **kwargs):
        batch = min(self.batch_size, self.num_rows - nskip)
        chunk = self.reader(
            self.file_path,
            nrows=batch,
            skiprows=nskip,
            names=self.names,
            header=False,
            sep=self.sep,
        )

        if columns:
            for col in columns:
                chunk[col] = chunk[col].astype(self.dtype[col])
            return chunk[columns]
        return chunk


#
# GPUFileIterator (Single File Iterator)
#


class GPUFileIterator:
    def __init__(
        self,
        file_path,
        engine=None,
        gpu_memory_frac=0.5,
        batch_size=None,
        columns=None,
        use_row_groups=None,
        dtype=None,
        names=None,
        row_size=None,
        **kwargs
    ):
        self.file_path = file_path
        self.engine = _get_read_engine(
            engine,
            file_path,
            batch_size=batch_size,
            gpu_memory_frac=gpu_memory_frac,
            use_row_groups=use_row_groups,
            dtype=dtype,
            names=names,
            row_size=None,
            **kwargs
        )
        self.columns = columns
        self.file_size = self.engine.num_rows
        self.rows_processed = 0
        self.cur_chunk = None
        self.count = 0

    def __iter__(self):
        self.rows_processed = 0
        self.count = 0
        self.cur_chunk = None
        return self

    def __len__(self):
        add_on = 0 if self.file_size % self.engine.batch_size == 0 else 1
        return self.file_size // self.engine.batch_size + add_on

    def __next__(self):
        if self.rows_processed >= self.file_size:
            if self.cur_chunk:
                chunk = self.cur_chunk
                self.cur_chunk = None
                return chunk
            raise StopIteration
        self._load_chunk()
        chunk = self.cur_chunk
        self.cur_chunk = None
        return chunk

    def _load_chunk(self):
        # retrieve missing final chunk from fileset,
        # will fail on last try before stop iteration in __next__
        if self.rows_processed < self.file_size and not self.cur_chunk:
            self.cur_chunk = self.engine.read_file_batch(
                nskip=self.rows_processed, columns=self.columns
            )
            self.count = self.count + 1
            self.rows_processed += self.cur_chunk.shape[0]


#
# GPUDatasetIterator (Iterates through multiple files)
#


class GPUDatasetIterator:
    def __init__(self, paths, **kwargs):
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            raise TypeError("paths must be a string or a list.")
        if len(paths) < 1:
            raise ValueError("len(paths) must be > 0.")
        self.paths = paths
        self.num_paths = len(paths)
        self.kwargs = kwargs
        self.itr = None
        self.next_path_ind = 0

    def __iter__(self):
        self.itr = None
        self.next_path_ind = 0
        return self

    def __next__(self):

        if self.itr is None:
            self.itr = GPUFileIterator(self.paths[self.next_path_ind], **self.kwargs)
            self.next_path_ind += 1

        while True:
            try:
                return self.itr.__next__()
            except StopIteration:
                if self.next_path_ind >= self.num_paths:
                    raise StopIteration
                path = self.paths[self.next_path_ind]
                self.next_path_ind += 1
                self.itr = GPUFileIterator(path, **self.kwargs)
