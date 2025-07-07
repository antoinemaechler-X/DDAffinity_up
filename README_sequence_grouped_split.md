# Sequence-Grouped Train/Val/Test Split

This document describes the new sequence-grouped train/val/test split implementation that addresses data leakage issues in protein-protein interaction prediction.

## Problem

The original train/val/test split was performed at the **complex level** (individual protein-protein interactions). However, many complexes with different names are actually very similar structurally, leading to data leakage where similar proteins appear in both training and test sets.

## Solution

The new implementation groups complexes by **sequence identity** (>70%) before performing the split. This ensures that highly similar protein complexes are kept together in the same split, preventing data leakage.

## Key Features

- **Sequence-based grouping**: Complexes with >70% sequence identity are grouped together
- **Group-level splitting**: Train/val/test split is performed on groups, not individual complexes
- **Multi-chain support**: Handles protein complexes with multiple chains
- **Robust sequence extraction**: Extracts actual protein sequences from PDB files using BioPython
- **Caching**: Groups are cached to avoid repeated computation
- **Drop-in replacement**: Compatible with existing training pipelines

## Files

### Main Implementation
- `rde/utils/skempi_mpnn_sequence_grouped.py` - Main dataset manager and dataset classes

### Training Script
- `train_DDAffinity_grouped.py` - Training script using sequence-grouped splits

### Test Script
- `test_sequence_grouped_split.py` - Test script to validate the implementation

### Example Usage
- `run_grouped_training_example.sh` - Example shell script showing different usage patterns

## Usage

### Basic Usage

```python
from rde.utils.skempi_mpnn_sequence_grouped import SequenceGroupedSkempiDatasetManager

# Create dataset manager
dataset_mgr = SequenceGroupedSkempiDatasetManager(
    config=config,
    num_cvfolds=1,  # Single fold since grouping provides separation
    num_workers=4,
    logger=logger
)

# Get data loaders
train_loader = dataset_mgr.get_train_loader(0)
val_loader = dataset_mgr.get_val_loader(0)
```

### Integration with Training Scripts

To use this in your training scripts, simply replace:

```python
# Old import
from rde.utils.skempi_mpnn import SkempiDatasetManager

# New import
from rde.utils.skempi_mpnn_sequence_grouped import SequenceGroupedSkempiDatasetManager
```

And update the dataset manager creation:

```python
# Old
dataset_mgr = SkempiDatasetManager(config, num_cvfolds=10, num_workers=4, logger=logger)

# New
dataset_mgr = SequenceGroupedSkempiDatasetManager(config, num_cvfolds=1, num_workers=4, logger=logger)
```

### Training

Use the dedicated training script for sequence-grouped splits:

```bash
# Normal training (uses cached sequence groups)
python train_DDAffinity_grouped.py configs/train/mpnn_ddg_simple.yml \
    --logdir ./logs_skempi_grouped \
    --early_stoppingdir ./early_stopping_grouped \
    --device cuda \
    --num_workers 4

# Training with reset_split (recomputes sequence groups)
python train_DDAffinity_grouped.py configs/train/mpnn_ddg_simple.yml \
    --logdir ./logs_skempi_grouped \
    --early_stoppingdir ./early_stopping_grouped \
    --device cuda \
    --num_workers 4 \
    --reset_split
```

### Testing

Run the test script to validate the implementation:

```bash
# Basic test
python test_sequence_grouped_split.py configs/train/mpnn_ddg.yml

# Compare with original split
python test_sequence_grouped_split.py configs/train/mpnn_ddg.yml --compare
```

## Configuration

The sequence-grouped split uses the same configuration files as the original implementation. Key parameters:

- `sequence_identity_threshold`: 0.9 (90% sequence identity for grouping)
- `train_ratio`: 0.7 (70% of groups for training)
- `val_ratio`: 0.15 (15% of groups for validation)
- `test_ratio`: 0.15 (15% of groups for testing)

### Training Parameters

- `--reset_split`: Force recomputation of sequence groups (ignores cache)
- `--logdir`: Directory for training logs (default: `./logs_skempi_grouped`)
- `--early_stoppingdir`: Directory for model checkpoints (default: `./early_stopping_grouped`)

## How It Works

1. **Sequence Extraction**: Extract protein sequences from PDB files for each complex
2. **Similarity Computation**: Compute pairwise sequence identities using global alignment
3. **Grouping**: Cluster complexes with >90% sequence identity into groups
4. **Caching**: Save groups to avoid repeated computation (unless `--reset_split` is used)
5. **Splitting**: Split groups into train/val/test sets (70/15/15)
6. **Entry Selection**: Select all entries from complexes in the appropriate groups

## Advantages

- **Prevents data leakage**: Similar proteins stay in the same split
- **More realistic evaluation**: Test set contains truly novel protein structures
- **Better generalization**: Model learns from diverse protein families
- **Maintains data integrity**: No complex-level leakage between splits
- **Efficient**: Groups are cached after first computation

## Performance Considerations

- **Initial computation**: Sequence extraction and similarity computation takes time on first run
- **Caching**: Results are cached in `sequence_groups.pkl` for subsequent runs
- **Memory usage**: Similarity matrix computation requires O(n²) memory for n complexes
- **Parallelization**: Sequence extraction can be parallelized for faster processing

## Example Output

```
Using cached sequence groups...

Sequence-based split summary:
Total complexes: 344
Total groups: 127
Train groups: 89, Val groups: 19, Test groups: 19
Train split: 6491 entries from 242 complexes
Val split: 502 entries from 35 complexes
Test split: 495 entries from 35 complexes

✓ No data leakage detected!
```

## Troubleshooting

### Common Issues

1. **PDB files not found**: Ensure PDB files are in the correct directory structure
2. **BioPython import errors**: Install BioPython: `pip install biopython`
3. **Memory issues**: Reduce batch size or number of workers
4. **Slow performance**: First run will be slower due to sequence extraction
5. **Parsing errors**: Some PDB files may have formatting issues - these are handled gracefully

### Debug Mode

Enable debug output by setting the logger level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Reset Cache

To force recomputation of groups (e.g., after changing threshold):

```python
# Pass reset=True to force recomputation
dataset = SequenceGroupedSkempiDataset(..., reset=True)
```

## Future Improvements

- **Structural similarity**: Add TM-score or RMSD-based grouping
- **Hierarchical clustering**: Use more sophisticated clustering algorithms
- **Dynamic thresholds**: Adjust similarity threshold based on dataset characteristics
- **Caching optimization**: Improve caching strategy for large datasets 