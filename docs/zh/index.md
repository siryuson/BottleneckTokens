# BToks 技术文档

> **版本**: 0.1
> **语言**: 中文

**统一的多模态嵌入框架，用于在图像、视频和文档上训练视觉-语言模型。**

---

## 文档导航

### 🏗️ 架构设计文档

深入了解 BToks 的系统架构和设计决策。

| 文档 | 说明 |
|------|------|
| [架构总览](./architecture/overview.md) | 整体架构、设计哲学、技术栈 |
| [模块管道系统](./architecture/module-pipeline.md) | 模块设计、features 字典、自定义模块 |
| [注册表系统](./architecture/registry-system.md) | Auto* 类、组件注册机制 |
| [配置系统](./architecture/config-system.md) | OmegaConf、继承、验证 |
| [训练系统](./architecture/training-system.md) | Trainer、分布式、检查点 |

### 📊 数据集文档

数据集溯源、格式规范、开发指南。

| 文档 | 说明 |
|------|------|
| [数据集总览](./datasets/index.md) | 所有训练/评测数据集索引、溯源卡片入口 |
| [格式规范](./datasets/format-spec.md) | 训练/评测数据集格式标准、接入检查清单 |
| [数据处理流水线](./datasets/architecture-data-pipeline.md) | 数据集、预处理、Collator |
| [自定义数据集](./datasets/guide-custom-dataset.md) | 数据集开发、Schema、Collator |
| [评测数据集 Lance 布局](./datasets/design-eval-lance-layout.md) | 评测数据集的 Lance 目录结构设计 |

### 📖 API 参考

详细的类和函数参考文档。

| 文档 | 说明 |
|------|------|
| [BToks Model](./api/model.md) | 核心模型类、Config、create_model |
| [Trainer](./api/trainer.md) | 训练器 API |

### 👨‍💻 开发者指南

为框架贡献者和扩展开发者准备的指南。

| 文档 | 说明 |
|------|------|
| [贡献指南](./developer-guide/contributing.md) | 开发环境、编码规范、测试 |

### 🚀 用户指南

帮助新用户快速上手的指南。

| 文档 | 说明 |
|------|------|
| [快速入门](./guides/getting-started.md) | 安装、环境配置、第一次训练 |

---

## 核心特性

- **模块管道架构**：对标 sentence-transformers，模块化组合
- **HuggingFace 集成**：继承 PreTrainedModel，支持 from_pretrained
- **基于注册表的组件**：通过 Auto* 装饰器自动发现和注册
- **YAML 配置**：继承、验证、参数插值
- **多数据集支持**：MMEB、ViDoRe、VisRAG、LLaVAHound
- **分布式训练**：DeepSpeed 集成

---

## 快速链接

### 常用任务

- **推理**: 参见 [BToks.forward()](./api/model.md#forward)
- **训练**: 参见 [架构总览](./architecture/overview.md)
- **自定义模块**: 参见 [模块管道系统](./architecture/module-pipeline.md)

### 核心概念

- **模块管道**: 模型由多个模块顺序组成，通过 features 字典通信
- **注册表系统**: 所有组件通过 Auto* 注册表管理
- **配置驱动**: YAML 配置控制所有行为，支持继承

### 设计参考

- [sentence-transformers](https://www.sbert.net/) - 模块设计参考
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - 生态集成

---

## 相关链接

- 公开仓库
- English Documentation: 使用页面右上角的语言切换器
