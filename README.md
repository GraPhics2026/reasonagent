# ReasonGenPilot

无训练的图像生成与假设性编辑 Agent 系统。三条通路 + 一个 MLLM Router + 统一 CLI / Web 入口。

| 通路 | 输入 | 目标 |
| --- | --- | --- |
| `gen` | 纯文本 prompt | 优化 prompt，并生成更符合描述的图像 |
| `edit` | 原图 + 假设性指令 | 推理反事实变化，指令式编辑并 VQA 验证 |
| `hybrid` | 原图 + 假设性指令（整图重构） | Reason 展开 `scene_prompt`，再走 gen 文生图 |

## 状态

| 通路 / 模块 | 责任人 | 状态 |
| --- | --- | --- |
| gen 通路（基座 + GenPilot 优化循环） | 梁爽 | ✅ |
| edit 通路（Reason Agent + Qwen-Image 指令编辑 + VQA refine） | 陈颖妍 | ✅ |
| hybrid 通路（scene_prompt 7 层后处理） | 苏香广 | ✅ |
| Router + 统一入口 + Gradio Demo + 答辩文档 | 艾博显 | ✅ |

测试：**`pytest reason/` → 67 passed**（44 hybrid + 23 router）。

## 目录结构

```text
ReasonGenPilot/
├── README.md
├── pipeline.py             # 统一 CLI 入口
├── demo_gradio.py          # Gradio Web Demo
├── requirements.txt
├── config/
│   └── .env.example
├── prompts/
│   ├── gen_system.txt      # gen prompt 优化 (legacy, 现已内联)
│   ├── reason_system.txt   # edit / hybrid Reason Agent
│   ├── edit_candidate.txt  # edit 多候选生成
│   ├── edit_refine.txt     # edit refine
│   └── router_system.txt   # Router MLLM 分类器
├── reason/
│   ├── api_client.py       # OpenAI-compatible MLLM (text + vision)
│   ├── t2i_client.py       # T2I 抽象: dry_run / dashscope / siliconflow / comfyui
│   ├── edit_client.py      # DashScope Qwen-Image 指令编辑
│   ├── reason_agent.py     # Reason Agent (mode = edit / hybrid)
│   ├── gen_pipeline.py     # gen 通路
│   ├── edit_pipeline.py    # edit 通路
│   ├── edit_verify_loop.py # edit 薄封装
│   ├── hybrid_pipeline.py  # hybrid 通路 (含 7 层 scene_prompt 后处理)
│   ├── router.py           # 三分流 Router (MLLM-only)
│   ├── schemas.py          # 共用 dataclass
│   ├── test_hybrid_pipeline.py
│   └── test_router.py
├── scripts/
│   └── run_comparison.py   # hybrid vs edit vs gen 对比实验
├── data/
│   └── input/              # 测试图片 + 案例 JSONL
└── docs/                   # 详细文档与答辩材料 (见下表)
```

## 环境准备

```bash
cp config/.env.example config/.env       # 填入 API key
pip install -r requirements.txt
```

`config/.env` 已加入 `.gitignore`，**不要提交真实密钥**。

### 配置示例（DashScope + 米莫 MLLM）

```env
# MLLM (Reason Agent / VQA / Router 共用)
MLLM_API_KEY=...
MLLM_BASE_URL=https://api.xiaomimimo.com/v1
MLLM_MODEL=mimo-v2.5

# T2I (gen / hybrid 用)
T2I_BACKEND=dashscope
T2I_API_KEY=...
T2I_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
T2I_MODEL=qwen-image-2.0-pro

# Edit (edit 通路用，可与 T2I 同 key)
EDIT_BACKEND=dashscope
EDIT_API_KEY=...
EDIT_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
EDIT_MODEL=qwen-image-2.0-pro
```

## 运行方式

### 统一入口 `pipeline.py`（推荐）

**单一 `--prompt` 字段对所有 mode**（gen 当描述读，edit / hybrid 当反事实指令读）。

```bash
# auto 模式 — Router 自动分流（无图 → gen；有图 → MLLM 决定 edit / hybrid）
python pipeline.py --prompt "A windmill in a poppy field" --output data/output/demo

# 强制 mode
python pipeline.py --mode edit \
  --image data/input/edit/elephant_squirrel_grass.png \
  --prompt "如果大象和松鼠玩跷跷板会怎样" \
  --output data/output/demo/seesaw \
  --real-api

python pipeline.py --mode hybrid \
  --image data/input/hybrid/bouquet.png \
  --prompt "如果这一束花分成两束独立的玫瑰，会是什么样子？" \
  --output data/output/demo/bouquet \
  --real-api
```

不带 `--real-api` 即 dry-run（SVG 占位图，无 API 消耗）。

### Gradio Web Demo

```bash
pip install gradio
python demo_gradio.py
# 默认 http://127.0.0.1:7860
```

UI 字段：Route 下拉 + 单一 Prompt 输入框 + 可选上传图 + Use real API 开关。
输出：before / after 图、推理链、VQA 结果、完整 JSON。

### 单独跑某条通路

```bash
python -m reason.gen_pipeline    --prompt "..." --output X --real-api
python -m reason.edit_pipeline   --image Y --instruction "..." --output X --real-api
python -m reason.hybrid_pipeline --image Y --instruction "..." --output X --real-api
```

### hybrid 对比实验

```bash
python scripts/run_comparison.py --real-api           # 全部 4 个用例
python scripts/run_comparison.py --real-api --cases 1,2  # 指定用例
```

## Router 设计要点

`reason/router.py` 是 **MLLM-only** 三分流器，**无关键词降级**：

| 层 | 行为 | 失败动作 |
| --- | --- | --- |
| 0 | `prompt` 必传（所有 mode）| `ValueError: router: prompt is required...` |
| 1 | `mode` 必须 ∈ `{auto, gen, edit, hybrid}` | `ValueError: router: unknown mode...` |
| 2 | 手动 mode + 输入兼容性（edit/hybrid 必须有可读的 image） | `ValueError: router: edit mode requires an image / image not found...` |
| 3 | auto + 无图 → 直接 `gen` | — |
| 4 | auto + 有图 → 调 MLLM（`prompts/router_system.txt`）| `RuntimeError: router: MLLM is not configured / call failed / unparseable JSON / invalid route...` |

错误统一两类：**`ValueError`**（用户输入问题）+ **`RuntimeError`**（环境/系统问题），所有信息以 `router: ` 前缀，便于日志聚合。

## 返回结构

三条 pipeline 均返回 `dataclass`，`.to_dict()` 序列化。**共同字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `final_image` | `str` | 最终输出图路径 |
| `final_prompt` | `str` | 最终使用的 prompt |
| `route` | `"gen" \| "edit" \| "hybrid"` | UI 分流展示用 |
| `reasoning_chain` | `list[str]` | 推理链（gen 为空） |
| `metadata` | `dict` | 运行参数 |

`edit` / `hybrid` 额外含：`image_before`、`instruction`、`reasoning_type`、`visual_cues`、`vqa_checklist`、`vqa_result` 等。

## 测试

```bash
pytest reason/                # 67 passed in 0.05s
pytest reason/test_router.py  # 23 passed (Router 单测)
pytest reason/test_hybrid_pipeline.py  # 44 passed (hybrid 7 层后处理)
```

## 与 ReasonBrain 论文的对比

参考 [ReasonBrain (2025)](https://arxiv.org/abs/2507.01908) 的假设性图像编辑（HI-IE）任务与四类推理：

| ReasonBrain | ReasonGenPilot |
|---|---|
| Reason50K 训练数据 | 零样本 MLLM prompt（`reason_system.txt`） |
| FRCE / CME 模块 | `visual_cues` + `physics_implications` + `preserve_objects` |
| FLUX 扩散端到端 | DashScope Qwen-Image 指令编辑 |
| 一次 forward | 多候选 + VQA + refine 迭代 |

`edit` 默认用指令编辑（**无 mask**），实验对比表明对反事实场景更稳定（详见 `docs/hybrid对比实验.md`）。

## 文档索引

| 文件 | 内容 |
|------|------|
| [docs/对接说明.md](./docs/对接说明.md) | 各模块 API 对接细节、统一入口模板、Gradio 集成代码 |
| [docs/报告.md](./docs/报告.md) | 完整报告 |
| [docs/report.md](./docs/report.md) | 完整报告英文版 |
| [docs/答辩.pptx](./docs/答辩.pptx) | 答辩 PPT（15 页） |
| [docs/hybrid对比实验.md](./docs/hybrid对比实验.md) | hybrid vs edit vs gen 对比实验 |
| [docs/ReasonGenPilot_开发计划.md](./docs/ReasonGenPilot_开发计划.md) | 项目精简版开发计划 |
| [docs/ReasonGenPilot_四人分工.md](./docs/ReasonGenPilot_四人分工.md) | 成员分工与交付清单 |
