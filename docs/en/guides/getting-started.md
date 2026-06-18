# Getting Started

This guide helps you install BToks and run your first experiment.

## Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (Compute Capability ≥7.0, Volta or newer)
- 16GB+ GPU memory recommended for 7B models

## Installation

### Standard Installation

```bash
# Install the package
pip install -e .
```

### Development Installation

For contributors and developers:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from vlm2emb import Config, Registry; print('BToks installed successfully!')"
```

### Full Installation

Install all optional dependencies:

```bash
pip install -e ".[full]"
```

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| transformers | ≥4.52.0 | Hugging Face transformers |
| accelerate | ≥0.20.0 | Distributed training |
| peft | ≥0.10.0 | Parameter-efficient fine-tuning |

## Quick Start

### 1. Verify Installation

```python
from vlm2emb import AutoModule

# Check available components
print("Available Modules:", AutoModule.list_modules())
```

### 2. Load Configuration

```python
from vlm2emb import load_config

# Load a public training preset
config = load_config("configs/presets/btoks_qwen2vl_2b_v1.yaml")

# Access configuration values
print(f"Model type: {config.model.type}")
```

### 3. Create Components from Configuration

```python
from vlm2emb import AutoModule

# Build module from config dict (must contain 'type' key)
backbone = AutoModule.from_config({
    "type": "Qwen2VLBackbone",
    "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
})

# Or use build method with explicit type
pooling = AutoModule.build("LastTokenPooling")

print(f"Hidden size: {backbone.hidden_size}")
```

## Project Structure

```
BToks/
├── configs/                 # YAML configuration files
│   ├── models/             # Model configurations
│   ├── peft/               # LoRA configurations
│   ├── training/           # Training strategies
│   ├── datasets/           # Dataset configurations
│   ├── eval/               # Evaluation fragments
│   └── presets/            # Complete experiment presets
├── src/vlm2emb/            # Source code
│   ├── auto.py             # Auto* registries
│   ├── config.py           # Configuration loading
│   ├── registry.py         # Registry implementation
│   ├── modules/            # Backbone, Pooling, Normalize
│   ├── data/               # Datasets, collators, preprocessing
│   └── training/           # Trainers, callbacks
├── docs/                   # Documentation
│   ├── en/                 # English documentation
│   └── zh/                 # Chinese documentation
└── tests/                  # Test suite
```

## Next Steps

- [Architecture Overview](../architecture/overview.md) - Understand the system design
- [Configuration Guide](../architecture/config-system.md) - Master YAML configurations
- [Development Guide](../developer-guide/contributing.md) - Contribute to BToks

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're using Python 3.11+:

```bash
python --version  # Should be 3.11.x or higher
```

### CUDA Issues

Verify CUDA is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Memory Issues

For large models, consider:
- Using LoRA for parameter-efficient fine-tuning
- Reducing batch size
- Enabling gradient checkpointing

---

_Need help? Open an issue in the public repository._
