"""Verify BToks dataset configuration.

Usage:
    python examples/datasets/verify_btoks_config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm2emb.auto import AutoDataset
from vlm2emb.config import load_config


def main():
    # Load BToks dataset configuration
    print("Loading BToks dataset configuration from configs/datasets/btoks_mmeb_train.yaml\n")
    datasets_config = load_config("configs/datasets/btoks_mmeb_train.yaml")

    # Load combined dataset
    combined_config = {
        "type": "combined",
        "datasets": datasets_config
    }
    combined_dataset = AutoDataset.from_config(combined_config)

    # Extract dataset information
    dataset_info = []
    total_weight = 0

    for i, name in enumerate(combined_dataset.names):
        dataset_config = datasets_config.get(name, {})
        if hasattr(dataset_config, 'get'):
            weight = combined_dataset.weights[i]
            dataset_type = dataset_config.get("type", "unknown")
            sample_count = combined_dataset.dataset_lengths[i]
            metadata = dataset_config.get("metadata", {})
            ntp_side = metadata.get("ntp_side", "none") if hasattr(metadata, "get") else "none"

            dataset_info.append({
                "name": name,
                "type": dataset_type,
                "weight": weight,
                "sample_count": sample_count,
                "ntp_side": ntp_side
            })
            total_weight += weight

    # Sort by weight (descending)
    dataset_info.sort(key=lambda x: x["weight"], reverse=True)

    # Print summary
    print(f"{'='*110}")
    print(f"BToks CONFIGURATION VERIFICATION")
    print(f"{'='*110}\n")
    print(f"Total Datasets: {len(dataset_info)}")
    print(f"Total Weight: {total_weight}")
    print(f"Total Samples: {sum(info['sample_count'] for info in dataset_info):,}\n")

    # Print dataset table
    print(f"{'Dataset Name':<30} {'Type':<25} {'W':>4} {'%':>6} {'Samples':>10} {'Expected':>12} {'NTP':>8}")
    print(f"{'-'*110}")

    max_expected = 0
    for info in dataset_info:
        percentage = (info["weight"] / total_weight) * 100
        sample_count = info["sample_count"]
        expected_length = int(sample_count * total_weight / info["weight"])

        max_expected = max(max_expected, expected_length)

        print(f"{info['name']:<30} {info['type']:<25} {info['weight']:>4} {percentage:>5.1f}% "
              f"{sample_count:>10,} {expected_length:>12,} {info['ntp_side']:>8}")

    print(f"{'-'*110}")
    print(f"{'TOTAL':<30} {'':<25} {total_weight:>4} {100.0:>5.1f}% "
          f"{sum(info['sample_count'] for info in dataset_info):>10,}")

    print(f"\n{'='*110}")
    print(f"CONSTRAINTS VERIFICATION")
    print(f"{'='*110}\n")
    print(f"✓ Total Weight = {total_weight} (target: 100)")
    print(f"✓ Max Expected Length = {max_expected:,} (target: <= 4,500,000)")
    print(f"✓ All weights are integers >= 1: {all(info['weight'] >= 1 and isinstance(info['weight'], (int, float)) for info in dataset_info)}")

    # Check constraints
    constraints_met = (
        total_weight == 100 and
        max_expected <= 4_500_000 and
        all(info['weight'] >= 1 for info in dataset_info)
    )

    print(f"\n{'✅ All constraints satisfied!' if constraints_met else '❌ Some constraints not satisfied'}")

    # NTP side statistics
    print(f"\n{'='*110}")
    print(f"NTP SIDE DISTRIBUTION")
    print(f"{'='*110}\n")

    ntp_stats = {}
    for info in dataset_info:
        side = info['ntp_side']
        if side not in ntp_stats:
            ntp_stats[side] = {'count': 0, 'weight': 0}
        ntp_stats[side]['count'] += 1
        ntp_stats[side]['weight'] += info['weight']

    for side, stats in sorted(ntp_stats.items()):
        print(f"{side:>8}: {stats['count']:>2} datasets, total weight = {stats['weight']:>3}")


if __name__ == "__main__":
    main()
