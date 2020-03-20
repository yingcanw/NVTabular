nvTabular
==========

Recommender systems require massive datasets to train, particularly for deep learning based solutions.  The transformation of these datasets after ETL in order to prepare them for model training is particularly challenging.  Often the time taken to do steps such as categorical encoding and normalization exceed the time it takes to train a model.

nvTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems.  It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.

The library is designed to be interoperable with both PyTorch and Tensorflow using batch dataloaders that we’ve developed as extensions of native framework code.  nvTabular provides the option to shuffle data during preprocessing, allowing the dataloader to load large contiguous chunks from files rather than individual elements.  We have benchmarked our dataloader at 100x the baseline item by item PyTorch dataloader and 3x the Tensorflow batch dataloader, with several optimizations yet to come in that stack.

Extending beyond model training, we plan to provide integration with model serving frameworks like NVidia’s Triton Inference Server, creating a clear path to production inference for these models and allowing the feature engineering and preprocessing steps performed on the data during training to be easily and automatically applied to incoming data during inference.

nvTabular is designed to support Data Scientists and ML Engineers trying to train deep learning recommender systems or other tabular data problems by allowing them to:
Prepare datasets quickly and easily in order to experiment and train more models.
Work with datasets that exceed GPU and CPU memory without having to worry about scale.
Think about what they want to do with the data, not how they have to do it, using our abstraction at the operation level.

It is also meant to help ML Ops Engineers deploying models into production by providing:
Integration with model serving frameworks like NVidia’s Triton Inference Server to make model deployment easy.
Faster dataset transformation, allowing for production models to be trained more frequently and kept up date helping improve responsiveness and model performance.

Our goal is faster iteration on massive tabular datasets, both for experimentation during training, and also for production model responsiveness.  

To be clear, this is an early alpha release, and we have a long way to go.  We have a working framework, but our operation set is extremely limited and every day we’re developing new optimizations that help improve the performance of the library.  If you’re interested in working with us to help develop this library we’re looking for early collaborators and contributors.  In the coming months we’ll be optimizing existing operations, adding a full set of common feature engineering and preprocessing operations, and extending our backend to support multi-node and multi-gpu systems.  Please reach out via email or see our guide on contributions.  We are particularly interested in contributions or feature requests for feature engineering or preprocessing operations that you have found helpful in your own workflows.

Getting Started:
----------------
nvTabular is available in the NVidia container repository at the following location:

[Docker quickstart]

Within the container is the codebase, along with all of our dependencies, particularly RAPIDS cuDF and rmm and a range of examples.  The easiest way to get started is to simply launch the container above and explore the examples within.  It is designed to work with Cuda 10.2.  As we mature more cuda versions will be supported.

[Pip/Conda Install]

If you wish to install the library yourself you can do so using the commands above.  The requirements for the library include:
Examples and Tutorials:

A workflow demonstrating the preprocessing and dataloading components of nvTabular can be found in the Joy of Cooking tutorial on training Facebook's Deep Learning Recommender Model (DLRM) on the Criteo 1TB dataset.

[ DLRM Criteo Workflow ]

We also have a simple tutorial that demonstrates similar functionality on a much smaller dataset, providing a pipeline for the Rossman store sales dataset fed into a fast.ai tabular data model.  

[ Rossman Store Sales ] 

Contributing
------------
If you wish to contribute to the library directly please see Contributing.md.  We are in particular interested in contributions or feature requests for feature engineering or preprocessing operations that you have found helpful in your own workflows.

Contributors
^^^^^^^^^^^^

nvTabular is supported and maintained directly by a small team of nvidians with a love of recommender systems.  They are: Julio Perez, Onur Yilmaz, Rick Zamora and Even Oldridge.

Learn More
----------

If you’re interested in learning more about how nvTabular works under the hood we have provided this more detailed description of the core functionality.

We also have API documentation that outlines in detail the specifics of the calls available within the library.

This is likely another page linked off of the main repo
How it works under the hood:

nvTabular wraps the RAPIDS cuDF library which provides the bulk of the functionality, accelerating dataframe operations on the GPU.  We found in our internal usage of cuDF on massive dataset like Criteo or RecSys 2020 that it wasn’t straightforward to use once the dataset had scaled past GPU memory.  The same design pattern kept emerging for us and we decided to package it up as nvTabular in order to make tabular data workflows simpler.

We provide mechanisms for iteration when the dataset exceeds GPU memory, allowing you to focus on what you want to do with your data, not how you need to do it.  We also provide a template for our core compute mechanism, Operations, or as we often refer to them ‘ops’ allowing you to build your own custom ops from cuDF and other libraries.

Follow our getting started guide to get nvTabular installed on your container or system.  Once installed you can setup a workflow in the following way:

[ setup example ]

With the workflow in place we can now explore the library in detail.
Operations:

Operations are a reflection of the way in which compute happens on the GPU across large datasets.  At a high level we’re concerned with two types of compute: the type that touches the entire dataset (or some large chunk of it) and the type that operates on a single row.  Operations split the compute such that the first phase, which we call statistics gathering, is the only place where operations that cross the row boundary can take place.  An example of this would be in the Normalize op which relies on two statistics, the mean and standard deviation.  In order to normalize a row, we must first have calculated these two values.

Statistics are further split into a chunked compute and a combine stage allowing for chunked iteration across datasets that don’t fit in GPU (or CPU) memory.  Where possible (and efficient) we utilize the GPU to do highly parallel compute, but many operations also rely on host memory for buffering and CPU compute when necessary.  The chunked results are combined to provide the statistics necessary for the next phase.

The second phase of operations is the apply phase, which uses the statistics created earlier to modify the dataset, transforming the data.  Notably we allow for the application of transforms not only during the modification of the dataset, but also during dataloading, with plans to support the same transforms during inference.

[ Underlying code for Normalize ]

In order to minimize iteration through the data we combine all of the computation required for statistics into a single computation graph that is applied chunkwise while the data is on GPU.  We similarly group the apply operation and transform the entire chunk at a time.  This lazy iteration style allows you to setup a desired workflow first, and then apply it to multiple datasets, including the option to apply statistics from one dataset to others.  Using this option the training set statistics can be applied to the validation and test sets preventing undesirable data leakage.

A higher level of abstraction:
nvTabular code is targeted at the operator level, not the dataframe level, providing a method for specifying the operation you want to perform, and the columns or type of data that you want to perform it on.

We make an explicit distinction between feature engineering ops, which we use to mean the creation of new variables, and preprocessing ops which transform data more directly to make it ready for the model to which it’s feeding.  While the type of computation involved in these two stages is often similar, we want to allow for the creation of new features that will then be preprocessed in the same way as other input variables.

Two main data types are supported; categorical variables and continuous variables.  Feature engineering operators explicitly take as input one or more continuous or categorical columns and produce one or more columns of a specific type.  By default the input columns used to create the new feature are also included in the output, however this can be overridden with the [replace] keyword in the operator.


[ Groupby example ] 

Preprocessing operators take in a set of columns of the same type and perform the operation across each column, transforming the output during the final operation into a long tensor in the case of categorical variables or a float tensor in the case of continuous variables.  Preprocessing operations replace the column values with their new representation by default, but again we allow the user to override this.

[ Normalization example ]

Operators may also be chained to allow for more complex feature engineering or preprocessing.  Chaining of operators is done by creating a list of the operators.  By default only the final operator in a chain that includes preprocessing will be included in the output with all other intermediate steps implicitly dropped.

[ Chaining example ] 

Framework Interoperability:
---------------------------

In addition to providing mechanisms for transforming the data to prepare it for deep learning models we also provide framework specific dataloaders to help optimize getting that data to the GPU.  Under a traditional dataloading scheme, data is read in item by item and collated into a batch.  PyTorch allows for multithreading to create multiple batches at the same time, while Tensorflow adopts a windowed buffering method to improve dataloading performance.  In PyTorch this still leads to many individual rows of tabular data accessed independently which impacts I/O, especially when this data is on the disk and not in CPU memory.  Tensorflow grabs larger chunks of data but only within a limited window of the dataset, meaning that the shuffle performed to randomize the data after that load isn’t uniformly distributed.

In nvTabular we provide an option to shuffle during dataset creation, allowing the dataloader to read in contiguous chunks of data that are already randomized across the entire dataset.  We provide the option to control the number of chunks that are combined into a batch, allowing the end user flexibility when trading off between performance and true randomization.

When compared to an item by item dataloader of PyTorch we have benchmarked our throughput as 100x faster dependent upon batch and tensor size.  Relative to Tensorflow we are ~2.5x faster with many optimizations still available.
