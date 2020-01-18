# ___PROJECT___ 0.0.0 (15 Jan 2020)

## New Features

- PR#11 add "sep" keyword support for GPUDatasetIterator for CSV data files
- PR#11 Added TransformOperator class for Feature Engineering phase
- PR#11 Added Feature Engineering phase to pipeline(see preproc.update_stats)
- PR#11 Added new datasets_ltm fixture for ltm tests (see test_dl_encoder.py)

## Improvements

- PR#11 dl_encoder refactor - supports Larger Than Memory categorical columns, no more nvstring/nvcategories


## Bug Fixes

- PR#11 pq_to_pq_processed fix to pass testing
- PR#11 Updated fixtures in tests, removed fixtures from preproc
