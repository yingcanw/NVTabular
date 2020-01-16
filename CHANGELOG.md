# ___PROJECT___ 0.0.0 (15 Jan 2020)

## New Features

- add "sep" keyword support for GPUDatasetIterator for CSV data files
- Added TransformOperator class for Feature Engineering phase
- Added Feature Engineering phase to pipeline(see preproc.update_stats)

## Improvements

- dl_encoder refactor - supports Larger Than Memory categorical columns, no more nvstring/nvcategories


## Bug Fixes

- pq_to_pq_processed fix to pass testing
- Updated fixtures in tests, removed fixtures from preproc
