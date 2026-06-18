# ViDoRe PDF

> ViDoRe 体系中的 PDF 文档检索训练子集。

| 属性 | 值 |
|------|-----|
| 模态 | visdoc |
| 任务类型 | `document_retrieval` |
| 用途 | 训练 |
| 溯源层数 | 2 层 |

## 原始数据集

### 来源信息
| 属性 | 值 |
|------|-----|
| 原始名称 | `vidore/colpali_train_set` (pdf subset) |
| 论文 | ColPali: Efficient Document Retrieval with Vision Language Models ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449)) |
| HuggingFace | [vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set) |
| 子集名 | pdf |
| 许可证 | 参见原始数据集 |

### Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| image | struct{bytes, path} | 文档页图像 |
| query | string | 查询文本 |
| answer | string | 正样本文本答案 |

### 样本

> 完整示例见下方“BToks Lance 格式”小节。

---

## BToks Lance 格式

### 来源信息
| 属性 | 值 |
|------|-----|
| 数据加载器 | `vidore_train` |
| 实现文件 | `src/vlm2emb/data/datasets/vidore.py` |
| 注册名 | `vidore_train` |
| Lance 路径 | `./data/converted` |
| 过滤条件 | `source = "pdf"` |

### Lance 布局

**train.lance（单表）**:

| 列名 | 类型 | 说明 |
|------|------|------|
| image | struct{bytes: binary, path: string} | 文档页图像 |
| image_filename | string | 原始图像文件名 |
| query | string | 查询文本 |
| answer | string | 正样本文本答案 |
| source | string | 来源子集 |
| options | string | 上游附加 metadata |
| page | string | 页码或页标识（若有） |
| model | string | 上游模型 metadata |
| prompt | string | 上游 prompt metadata |
| answer_type | string | 答案类型 metadata |

### 训练侧映射
- query: 运行时默认渲染为 `"<|image_pad|> {query}"` + 单页图像
- positive: `answer`（纯文本正样本）
- negative: 空文本（占位）
- `source / page / answer_type` 等字段当前保留为存储 metadata，不直接进入最小训练视图

### 样本

以下示例直接采样自 `./data/converted`，并限定 `source = "pdf"`。

- `image.bytes` 按 `bytes / sha1 / format / size` 展示摘要。
- 其余字段保留 Lance 原值。

#### 示例 1

```json
{
  "data/train.lance": {
    "image": {
      "path": "page_9.jpg",
      "bytes": {
        "bytes": 319580,
        "sha1": "855059eda212",
        "format": "JPEG",
        "size": [
          1700,
          2200
        ]
      }
    },
    "image_filename": "data/scrapped_pdfs_split/pages_extracted/energy_train/1d09a977-063b-463f-a897-2eda99c1a4f6.pdf/page_9.jpg",
    "query": "What is the duration of the course mentioned in the image?",
    "answer": "['five to ten hours, not including field trips']",
    "source": "pdf",
    "options": null,
    "page": "9",
    "model": "sonnet",
    "prompt": "\n        You are an assistant specialized in Multimodal RAG tasks.\n\n        The task is the following: given an image from a pdf page, you will have to \n        generate questions that can be asked by a user to retrieve information from \n        a large documentary corpus. \n        The question should be relevant to the page, and should not be too specific \n        or too general. The question should be about the subject of the page, and \n        the answer need to be found in the page. \n\n        Remember that the question is asked by a user to get some information from a\n        large documentary corpus that contains multimodal data. Generate a question \n        that could be asked by a user without knowing the existence and the content \n        of the corpus. \n\n        Generate as well the answer to the question, which should be found in the\n        page. And the format of the answer should be a list of words answering the\n        question. \n\n\n        Generate at most THREE pairs of questions and answers per page in a \n        dictionary with the following format, answer ONLY this dictionary\n        NOTHING ELSE: \n\n\n        {\n            \"questions\": [\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n            ]\n        }\n        where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers\n        that could be as long as needed. \n\n\n        Note: If there are no questions to ask about the page, return an empty list.\n        Focus on making relevant questions concerning the page. \n\n        Here is the page: \n\n",
    "answer_type": null
  }
}
```

#### 示例 2

```json
{
  "data/train.lance": {
    "image": {
      "path": "page_592.jpg",
      "bytes": {
        "bytes": 401271,
        "sha1": "79e1bb684c1e",
        "format": "JPEG",
        "size": [
          1700,
          2200
        ]
      }
    },
    "image_filename": "data/scrapped_pdfs_split/pages_extracted/energy_train/3794f052-3c8b-4890-afd2-49500c29ae26.pdf/page_592.jpg",
    "query": "How is the baseline CO2 emissions calculated for affected EGUs in the low load natural gas-fired or oil-fired subcategories?",
    "answer": "['by dividing the total CO2 emissions (in pounds) over the continuous time period by the total heat input (in MMBtu)']",
    "source": "pdf",
    "options": null,
    "page": "592",
    "model": "sonnet",
    "prompt": "\n        You are an assistant specialized in Multimodal RAG tasks.\n\n        The task is the following: given an image from a pdf page, you will have to \n        generate questions that can be asked by a user to retrieve information from \n        a large documentary corpus. \n        The question should be relevant to the page, and should not be too specific \n        or too general. The question should be about the subject of the page, and \n        the answer need to be found in the page. \n\n        Remember that the question is asked by a user to get some information from a\n        large documentary corpus that contains multimodal data. Generate a question \n        that could be asked by a user without knowing the existence and the content \n        of the corpus. \n\n        Generate as well the answer to the question, which should be found in the\n        page. And the format of the answer should be a list of words answering the\n        question. \n\n\n        Generate at most THREE pairs of questions and answers per page in a \n        dictionary with the following format, answer ONLY this dictionary\n        NOTHING ELSE: \n\n\n        {\n            \"questions\": [\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n                {\n                    \"question\": \"XXXXXX\",\n                    \"answer\": [\"YYYYYY\"]\n                },\n            ]\n        }\n        where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers\n        that could be as long as needed. \n\n\n        Note: If there are no questions to ask about the page, return an empty list.\n        Focus on making relevant questions concerning the page. \n\n        Here is the page: \n\n",
    "answer_type": null
  }
}
```
