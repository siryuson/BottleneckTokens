# Wiki-SS-NQ 训练数据集

> Wikipedia 截图问答检索数据集，用于训练文本到图像检索能力。

| 属性 | 值 |
|------|-----|
| 模态 | image |
| 任务类型 | `image_retrieval` |
| 用途 | 训练 / OOD 评测 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | Wiki-SS-NQ |
| 论文 | Wiki-SS-NQ: Wikipedia Screenshot Natural Questions |
| HuggingFace | [Tevatron/wiki-ss-nq](https://huggingface.co/datasets/Tevatron/wiki-ss-nq) |
| 本地 query 仓 | `./data/converted` |
| 本地 corpus 仓 | `./data/converted` |
| 许可证 | 参见上游数据集 |

### 本地审计结论

- `train.jsonl`: `44,002` 条 query
- `test.jsonl`: `3,610` 条 query
- screenshot corpus: `1,267,874` 张截图
- 当前 `MMEB-V2 / Wiki-SS-NQ` 的 `1000` 条 eval query 全部包含在本地 `train.jsonl` 中
- 当前 `MMEB-V2 / Wiki-SS-NQ` 的 `10,694` 个 eval candidate `docid` 与本地 train positive `docid` 无重叠

因此当前标准化根目录默认采取：

- `official_train_raw`: 保留本地原始训练 query-positive 对
- `official_train`: 过滤掉与 `MMEB-V2` eval query 文本重叠的 `1000` 条 query 后得到的安全训练视图

### 原始 Schema

`train.jsonl`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query_id` | string | query 标识 |
| `query` | string | 原始问题文本 |
| `positive_passages` | list[object] | 正样本截图 passage，包含 `docid/title/text` |
| `negative_passages` | list[object] | 负样本文本 passage，当前训练实现不直接消费 |
| `answers` | list[string] | 参考答案 |

`wiki-ss-corpus-new/train`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | PIL.Image / bytes | Wikipedia 截图 |
| `docid` | string | corpus 顺序号；训练 positive 使用的是 `image.path` 的文件名 stem |

---

## BToks 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `wikissnq_train` |
| 实现文件 | `src/vlm2emb/data/datasets/wikissnq_train.py` |
| 注册名 | `wikissnq_train` |
| 转换脚本 | `scripts/convert/train/convert_wikissnq_to_lance.py` |
| 标准化根目录 | `./data/converted` |

### Lance 结构

- `data/images.lance`
  - 共享 screenshot corpus
  - 字段为 `path: string`、`image: binary`
  - `path` 使用 `image.path` 的文件名 stem，与训练 `positive_docid` 对齐
- `data/official_train_raw.lance`
  - 一行对应一个 `query x positive_passage`
  - 当前会按 `positive_passages` 爆开
- `data/official_train.lance`
  - 过滤掉 `MMEB-V2` eval-overlap query 后的默认训练视图
- `data/official_train_without_mmeb_v2_eval.lance`
  - 与 `official_train` 等价，命名明确说明默认训练 split 已排除 MMEB-V2 eval query 重叠
- `data/exclusions/mmeb_v2_eval.lance`
  - 被过滤的 query-positive 对，用于审计，不作为默认训练输入

### 当前样本规模

- `official_train_raw`: `340,094`
- `official_train`: `332,518`
- `official_train_without_mmeb_v2_eval`: `332,518`
- `exclusions/mmeb_v2_eval`: `7,576`
- `images`: `1,267,874`

### 训练侧映射

- query: `Find the document image that can answer the given query: {query}`
- positive: `<|image_pad|>\nRepresent the given image`
- negative: 空文本（占位）

### Transform 参数

默认 transform 属于 `wikissnq_train` runtime。配置里的 `transform.query.*`、`transform.positive.*` 和 `transform.negative.*` 是默认行为的显式参数，不依赖旧 text-format 路径。

runtime 只负责 instruction/body 拼接、visual token placement 和尾部换行等模型可见 surface 规则；不会对原始 query 做 detokenize 或其它文本清洗。因此源 query 中若出现 `n 't` 这类上游 tokenization 痕迹，会按原文保留。

### 配置示例

```yaml
Wiki-SS-NQ:
  type: wikissnq_train
  path: ./data/converted
  dataset_name: Wiki-SS-NQ
  split: official_train_without_mmeb_v2_eval
  weight: 1
  transform:
    query:
      instruction: "Find the document image that can answer the given query:"
      instruction_body_separator: space
      trailing_newline: ensure_single
    positive:
      instruction: "Represent the given image"
      visual_token_placement: own_line
      trailing_newline: ensure_single
    negative:
      empty: empty_multimodal_input
```

### 样本

示例 query：

```text
Find the document image that can answer the given query: who founded the red cross to relieve the suffering of the war wounded
```

示例 positive：

```text
<|image_pad|>
Represent the given image
```

说明：

- query 不带图像
- positive 为一张 Wikipedia 截图
- `official_train` 默认已经排除了与 `MMEB-V2 / Wiki-SS-NQ` eval query 文本重叠的样本
