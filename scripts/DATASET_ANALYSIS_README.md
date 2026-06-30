# Academic Dataset Analysis Tool

Professional academic visualization tool for the Falcon Social Navigation Dataset.

## Features

This script generates three main academic visualizations in PDF format:

1. **Trajectory Length Distribution** - Shows the distribution of episode step counts across train/val sets
2. **Pedestrian Density Heatmap** - Displays pedestrian density across different scenes, highlighting high-density scenarios (avg. 6.8 persons/scene)
3. **Action Distribution Pie Chart** - Illustrates the proportion of movement actions (forward, turn, stop)

All visualizations are:
- Exported in **PDF format** for publication quality
- Rendered in **professional academic English**
- Styled with **Times New Roman** font
- Optimized for **academic papers and presentations**

## Requirements

```bash
pip install numpy matplotlib seaborn pandas tqdm
```

## Usage

### Basic Usage

```bash
python scripts/analyze_dataset_academic.py
```

This will analyze the default train/val datasets and save results to `dataset_analysis_academic/`.

### Custom Paths

```bash
python scripts/analyze_dataset_academic.py \
  --train_dir /path/to/train \
  --val_dir /path/to/val \
  --output_dir /path/to/output
```

### Quick Test

```bash
bash scripts/test_analysis.sh
```

## Output Files

The script generates the following files in the output directory:

### PDF Visualizations
- `trajectory_length_distribution.pdf` - Step length distribution for train/val sets
- `pedestrian_density_heatmap.pdf` - Heatmap showing pedestrian density across top 20 scenes
- `action_distribution.pdf` - Pie charts showing action distribution

### Data Files
- `dataset_statistics.txt` - Comprehensive statistics report
- `train_episodes.csv` - Training episode metadata
- `val_episodes.csv` - Validation episode metadata
- `all_episodes.csv` - Combined episode metadata

## Visualization Details

### 1. Trajectory Length Distribution
- **Format**: Histogram with mean/median lines
- **Metrics**: Mean, median, std, min, max trajectory lengths
- **Purpose**: Show the diversity of episode lengths in the dataset

### 2. Pedestrian Density Heatmap
- **Format**: 2D heatmap (5×4 grid)
- **Metrics**: Average pedestrian count per episode across scenes
- **Purpose**: Highlight high-density scenarios and spatial distribution challenges
- **Note**: Emphasizes the average 6.8 persons/scene density

### 3. Action Distribution
- **Format**: Pie chart with detailed statistics
- **Categories**: 
  - Move Forward
  - Turn (Left/Right combined)
  - Stop
- **Purpose**: Show the balance of navigation behaviors in the dataset

## Example Statistics Output

```
Falcon Social Navigation Dataset - Statistics Report
================================================================================

Train Dataset Statistics:
--------------------------------------------------------------------------------
  Total Episodes: 5,234
  Total Scenes: 72
  
Trajectory Length Statistics:
  Mean: 127.45 steps
  Median: 115.00 steps
  Std: 45.23 steps
  Min: 15 steps
  Max: 398 steps

Pedestrian Statistics:
  Episodes with pedestrians: 4,892 (93.5%)
  Average pedestrians per episode: 6.8 persons
  Max pedestrians in single frame: 15 persons

Action Distribution:
  Total actions: 667,123
  STOP (0): 5,234 (0.8%)
  MOVE_FORWARD (1): 534,234 (80.1%)
  TURN_LEFT (2): 63,845 (9.6%)
  TURN_RIGHT (3): 63,810 (9.6%)
```

## Dataset Structure Expected

The script expects the following directory structure:

```
data/collect_data/
├── train/
│   └── {scene_name}.basis/
│       └── {episode_id}/
│           ├── action/
│           │   └── 0.json          # Action sequence
│           ├── human_num/
│           │   └── 0.json          # Pedestrian counts per step
│           ├── pose/
│           │   └── 0.json          # Agent poses
│           └── rgb/
│               └── *.jpg           # RGB images
└── val/
    └── (same structure as train)
```

## Notes

- All text in visualizations is in **English only** (no Chinese characters)
- PDF files use **TrueType fonts** (fonttype 42) for better compatibility
- The script handles nested list formats in `human_num` data (e.g., `[[6], [6]]` → `[6, 6]`)
- Missing data is handled gracefully with warnings

## Troubleshooting

### Issue: "No pedestrian data available"
- Check that `human_num/0.json` files exist in episode directories
- Verify the JSON format is correct

### Issue: "No actions found"
- Check that `action/0.json` files exist
- Alternatively, the script will estimate steps from RGB image count

### Issue: PDF fonts look wrong
- Ensure matplotlib is properly installed with font support
- The script uses Times New Roman (serif) as default

## Citation

If you use this analysis tool in your research, please cite:

```bibtex
@inproceedings{falcon2024,
  title={Falcon: Social Navigation Dataset with Dynamic Pedestrians},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

## Contact

For issues or questions, please open an issue on the project repository.

