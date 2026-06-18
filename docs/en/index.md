# BToks Documentation

> **Version**: 0.1
> **Language**: English

**A Unified Multimodal Embedding Framework for Training Vision-Language Models on Images, Videos, and Documents.**

---

## Documentation Navigation

### 🏗️ Architecture Design

In-depth documentation on BToks's system architecture and design decisions.

| Document | Description |
|----------|-------------|
| [Architecture Overview](./architecture/overview.md) | Overall architecture, design philosophy, tech stack |
| [Module Pipeline System](./architecture/module-pipeline.md) | Module design, features dict, custom modules |
| [Registry System](./architecture/registry-system.md) | Auto* classes, component registration |
| [Configuration System](./architecture/config-system.md) | OmegaConf, inheritance, validation |
| [Training System](./architecture/training-system.md) | Trainer, distributed, checkpoints |

### 📊 Dataset Documentation

Dataset provenance, format specifications, and development guides.

| Document | Description |
|----------|-------------|
| [Dataset Overview](./datasets/index.md) | All training/evaluation dataset index, provenance card entry |
| [Format Specification](./datasets/format-spec.md) | Training/evaluation dataset format standards, onboarding checklist |
| [Data Pipeline](./datasets/architecture-data-pipeline.md) | Datasets, preprocessing, Collator |
| [Custom Dataset](./datasets/guide-custom-dataset.md) | Dataset development, Schema, Collator |
| [Eval Dataset Lance Layout](./datasets/design-eval-lance-layout.md) | Evaluation dataset Lance directory structure design |

### 📖 API Reference

Detailed class and function reference documentation.

| Document | Description |
|----------|-------------|
| [BToks Model](./api/model.md) | Core model class, Config, create_model |
| [Trainer](./api/trainer.md) | Trainer API |

### 👨‍💻 Developer Guide

Guides for framework contributors and extension developers.

| Document | Description |
|----------|-------------|
| [Contributing Guide](./developer-guide/contributing.md) | Dev environment, coding standards, testing |

### 🚀 User Guides

Guides to help new users get started quickly.

| Document | Description |
|----------|-------------|
| [Getting Started](./guides/getting-started.md) | Installation, environment setup, first training run |

---

## Core Features

- **Modular Pipeline Architecture**: Modular composition similar to sentence-transformers
- **HuggingFace Integration**: Inherits PreTrainedModel, supports from_pretrained
- **Registry-Based Components**: Auto* decorators for automatic discovery and registration
- **YAML Configuration**: Inheritance, validation, parameter interpolation
- **Multi-Dataset Support**: MMEB, ViDoRe, VisRAG, LLaVAHound
- **Distributed Training**: DeepSpeed integration

---

## Quick Links

### Common Tasks

- **Inference**: See [BToks.forward()](./api/model.md#forward)
- **Training**: See [Architecture Overview - Quick Start](./architecture/overview.md#6-quick-start)
- **Custom Modules**: See [Module Pipeline System - Custom Module Development](./architecture/module-pipeline.md#6-custom-module-development)

### Core Concepts

- **Module Pipeline**: Model consists of sequential modules communicating via features dict
- **Registry System**: All components managed through Auto* registries
- **Configuration-Driven**: YAML configuration controls all behavior, supports inheritance

### Design References

- [sentence-transformers](https://www.sbert.net/) - Module design reference
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Ecosystem integration

---

## Related Links

- Public repository
- [中文文档](/zh/)
