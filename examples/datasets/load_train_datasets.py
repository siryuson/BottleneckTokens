"""Load and inspect training dataset weights.

This example shows the weight configuration of each dataset in the mixed training set,
along with sample counts and expected traversal length.

Usage:
    python examples/datasets/load_train_datasets.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm2emb.auto import AutoDataset
from vlm2emb.config import load_config


def main():
    # Load dataset configuration
    print("Loading dataset configuration from configs/datasets/mmeb_train.yaml\n")
    datasets_config = load_config("configs/datasets/mmeb_train.yaml")
    
    # Load the combined dataset to get sample counts
    print("Loading combined dataset...\n")
    
    # Wrap config in combined dataset structure
    combined_config = {
        "type": "combined",
        "datasets": datasets_config
    }
    combined_dataset = AutoDataset.from_config(combined_config)
    
    # Extract information from combined dataset
    dataset_info = []
    total_weight = 0
    
    if hasattr(combined_dataset, 'dataset_lengths') and hasattr(combined_dataset, 'weights') and hasattr(combined_dataset, 'names'):
        # CombinedDataset with metadata
        for i, name in enumerate(combined_dataset.names):
            # Get config for this dataset
            dataset_config = datasets_config.get(name, {})
            if hasattr(dataset_config, 'get'):
                weight = combined_dataset.weights[i]
                dataset_type = dataset_config.get("type", "unknown")
                sample_count = combined_dataset.dataset_lengths[i]
                
                dataset_info.append({
                    "name": name,
                    "type": dataset_type,
                    "weight": weight,
                    "sample_count": sample_count
                })
                total_weight += weight
    else:
        # Fallback: single dataset or no metadata
        print("Warning: Dataset does not have metadata attributes (dataset_lengths, weights, names)")
        print("Showing config-based weights only\n")
        
        for dataset_name, dataset_config in datasets_config.items():
            if hasattr(dataset_config, 'get'):
                weight = dataset_config.get("weight", 1)
                dataset_type = dataset_config.get("type", "unknown")
                
                dataset_info.append({
                    "name": dataset_name,
                    "type": dataset_type,
                    "weight": weight,
                    "sample_count": None
                })
                total_weight += weight
    
    # Sort by weight (descending)
    dataset_info.sort(key=lambda x: x["weight"], reverse=True)
    
    # Print summary
    print(f"{'='*80}")
    print(f"Total Datasets: {len(dataset_info)}")
    print(f"Total Weight: {total_weight}")
    print(f"{'='*80}\n")
    
    # Print dataset table with sample counts and expected traversal length
    print(f"{'Dataset Name':<30} {'Type':<25} {'Weight':>8} {'%':>6} {'Samples':>10} {'Expected Len':>12}")
    print(f"{'-'*100}")
    
    total_samples = 0
    datasets_with_counts = 0
    
    for info in dataset_info:
        percentage = (info["weight"] / total_weight) * 100
        sample_count = info["sample_count"]
        
        # Calculate expected traversal length: samples / (weight / total_weight)
        # = samples * total_weight / weight
        if sample_count is not None:
            expected_length = int(sample_count * total_weight / info["weight"])
            total_samples += sample_count
            datasets_with_counts += 1
            print(f"{info['name']:<30} {info['type']:<25} {info['weight']:>8.1f} {percentage:>5.1f}% "
                  f"{sample_count:>10,} {expected_length:>12,}")
        else:
            print(f"{info['name']:<30} {info['type']:<25} {info['weight']:>8.1f} {percentage:>5.1f}% "
                  f"{'N/A':>10} {'N/A':>12}")
    
    print(f"{'-'*100}")
    print(f"{'TOTAL':<30} {'':<25} {total_weight:>8.1f} {100.0:>5.1f}% {total_samples:>10,}")
    print(f"\nDatasets with sample counts: {datasets_with_counts}/{len(dataset_info)}")
    print()
    
    # Group by type
    print(f"\n{'='*80}")
    print("Datasets Grouped by Type:")
    print(f"{'='*80}\n")
    
    type_groups = {}
    for info in dataset_info:
        dtype = info["type"]
        if dtype not in type_groups:
            type_groups[dtype] = []
        type_groups[dtype].append(info)
    
    for dtype, datasets in sorted(type_groups.items()):
        type_weight = sum(d["weight"] for d in datasets)
        type_samples = sum(d["sample_count"] for d in datasets if d["sample_count"] is not None)
        type_percentage = (type_weight / total_weight) * 100
        print(f"{dtype} (Weight: {type_weight:.1f}, {type_percentage:.1f}%, Samples: {type_samples:,}):")
        for d in datasets:
            sample_str = f"{d['sample_count']:,}" if d['sample_count'] is not None else "N/A"
            print(f"  - {d['name']:<35} weight={d['weight']:.1f}  samples={sample_str}")
        print()


if __name__ == "__main__":
    main()