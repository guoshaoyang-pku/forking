---
dataset_info:
  features:
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: ground_truth
    dtype: string
  - name: dataset
    dtype: string
  - name: constraint_type
    dtype: string
  - name: constraint
    dtype: string
  splits:
  - name: train
    num_bytes: 14723526.347425366
    num_examples: 7500
  download_size: 1947829
  dataset_size: 14723526.347425366
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
