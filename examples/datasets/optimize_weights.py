"""Optimize dataset weights and print inheritance-based YAML overrides."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm2emb.auto import AutoDataset
from vlm2emb.config import load_config, to_native_config


def calculate_expected_length(sample_count, total_weight, weight):
    """Calculate expected traversal length."""
    return int(sample_count * total_weight / weight)


def optimize_weights(dataset_info, target_total_weight=100, max_expected_len=4_500_000):
    """Optimize weights to meet constraints."""

    min_weights = []
    for info in dataset_info:
        sample_count = info["sample_count"]
        min_weight = max(1, int(sample_count * target_total_weight / max_expected_len) + 1)
        min_weights.append(min_weight)

    current_total = sum(info["weight"] for info in dataset_info)
    scale_factor = target_total_weight / current_total

    scaled_weights = []
    for info in dataset_info:
        scaled_weights.append(max(1, round(info["weight"] * scale_factor)))

    optimized_weights = []
    for index, _info in enumerate(dataset_info):
        optimized_weights.append(max(min_weights[index], scaled_weights[index]))

    current_sum = sum(optimized_weights)
    while current_sum != target_total_weight:
        diff = target_total_weight - current_sum
        if diff > 0:
            min_sample_idx = min(
                range(len(dataset_info)),
                key=lambda i: dataset_info[i]["sample_count"],
            )
            optimized_weights[min_sample_idx] += 1
            current_sum += 1
            continue

        best_idx = -1
        best_room = -1
        for index in range(len(dataset_info)):
            room = optimized_weights[index] - min_weights[index]
            if room > best_room:
                best_room = room
                best_idx = index

        if best_idx >= 0 and best_room > 0:
            optimized_weights[best_idx] -= 1
            current_sum -= 1
        else:
            break

    return optimized_weights


def main():
    """Load the base config, optimize weights, and print override YAML."""

    print("Loading dataset configuration...\n")
    datasets_config = to_native_config(
        load_config(REPO_ROOT / "configs/datasets/mmeb_train.yaml"),
        resolve=True,
    )

    combined_dataset = AutoDataset.from_config(
        {
            "type": "combined",
            "datasets": datasets_config,
        }
    )

    dataset_info = []
    for index, name in enumerate(combined_dataset.names):
        dataset_config = datasets_config.get(name, {})
        dataset_info.append(
            {
                "name": name,
                "type": dataset_config.get("type", "unknown"),
                "weight": combined_dataset.weights[index],
                "sample_count": combined_dataset.dataset_lengths[index],
            }
        )

    print("=" * 100)
    print("CURRENT STATE")
    print("=" * 100)
    current_total = sum(info["weight"] for info in dataset_info)
    print(f"Total Weight: {current_total}")
    print(f"Total Samples: {sum(info['sample_count'] for info in dataset_info):,}\n")

    print("Datasets with Expected Length > 5M:")
    for info in dataset_info:
        expected_len = calculate_expected_length(
            info["sample_count"],
            current_total,
            info["weight"],
        )
        if expected_len > 5_000_000:
            print(
                f"  {info['name']:<30} weight={info['weight']:>5.1f}  "
                f"samples={info['sample_count']:>7,}  expected={expected_len:>10,}"
            )

    print(f"\n{'=' * 100}")
    print("OPTIMIZED WEIGHTS (Target: Total=100, Max Expected Length=4.5M)")
    print("=" * 100)

    optimized_weights = optimize_weights(
        dataset_info,
        target_total_weight=100,
        max_expected_len=4_500_000,
    )

    for index, info in enumerate(dataset_info):
        info["optimized_weight"] = optimized_weights[index]

    dataset_info.sort(key=lambda item: item["optimized_weight"], reverse=True)

    print(
        f"\n{'Dataset Name':<30} {'Old W':>6} {'New W':>6} "
        f"{'Samples':>10} {'Expected Len':>12} {'Change':>8}"
    )
    print("-" * 100)

    total_optimized = sum(info["optimized_weight"] for info in dataset_info)
    max_expected = 0
    for info in dataset_info:
        old_weight = info["weight"]
        new_weight = info["optimized_weight"]
        sample_count = info["sample_count"]
        expected_len = calculate_expected_length(sample_count, 100, new_weight)
        change = ((new_weight / old_weight) - 1) * 100
        max_expected = max(max_expected, expected_len)
        print(
            f"{info['name']:<30} {old_weight:>6.1f} {new_weight:>6} "
            f"{sample_count:>10,} {expected_len:>12,} {change:>7.1f}%"
        )

    print("-" * 100)
    print(f"{'TOTAL':<30} {current_total:>6.1f} {total_optimized:>6}")
    print(f"\nMax Expected Length: {max_expected:,}")
    print(f"All constraints satisfied: {total_optimized == 100 and max_expected <= 4_500_000}")

    print(f"\n{'=' * 100}")
    print("YAML CONFIG (copy to configs/datasets/mmeb_train_optimized.yaml)")
    print("=" * 100)
    print()
    print("_inherit_: ./mmeb_train.yaml")
    print()

    optimized_by_name = {
        info["name"]: info["optimized_weight"]
        for info in dataset_info
    }
    for name in datasets_config:
        print(f"{name}:")
        print(f"  weight: {optimized_by_name[name]}")
        print()


if __name__ == "__main__":
    main()
