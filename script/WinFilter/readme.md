# WinFilter Experiments




### Environment Setup



Install the WinFilter library:

```bash
# Navigate to the algorithm directory
cd /data/HybridANNS/algorithm/RangeFilteredANN/

# Install the Python package in development mode
pip3 install .
```


## Running Experiments

The experiments script automatically runs the WinFilter methods on all available datasets with their corresponding filter widths.

### Available Datasets

- `deep`: 96-dimensional vectors with Euclidean distance
- `wit`: 2048-dimensional vectors with Euclidean distance
- `text2image`: 200-dimensional vectors with Euclidean distance
- `yt8mAudio`: 128-dimensional vectors with Euclidean distance

### Executing the Experiments

1. Navigate to the WinFilter script directory:

```bash
cd /data/HybridANNS/script/WinFilter/
```

2. Run the experiments script:

```bash
python3 run.py
```

3. To run experiments on specific datasets only:

```bash
python3 run.py --datasets deep wit
```

### Results

All experiment results will be saved to `/data/HybridANNS/data/Experiment/Result/WinFilter/` with the following structure:

- `/results/{dataset}/{filter_width}_{dataset}_results.csv`: Experiment results including recall, query time, and QPS
- `/build_times/{filter_width}_{dataset}_build_times.csv`: Index build times
- `/index_cache/{dataset}/`: Index cache files

## Methods Overview

The experiments evaluate several filtering methods:

1. **Prefiltering**: Filtering before graph traversal
2. **Postfiltering**: Standard graph traversal followed by filtering
3. **Optimized Postfiltering**: Improved postfiltering with efficiency optimizations
4. **Vamana Tree**: Hierarchical graph structure for range filtering
5. **Smart Combined**: Adaptive hybrid approach
6. **Three Split**: Three-way partition filtering method
7. **Super Optimized Postfiltering**: Advanced optimized filtering approach

## Troubleshooting

If you encounter any issues:

- Ensure all dependencies are correctly installed
- Check that the data paths are correct in the script
- Verify that the WinFilter library was installed successfully

