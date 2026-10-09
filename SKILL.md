---
name: image-gen
description: 通过可配置的图片 API 完成文生图、参考图生图和持续图片编辑，支持 OpenAI Images 与 Gemini generateContent 兼容接口。适用于根据提示词生成图片、参考已有图片创作新画面，或基于图片和会话修改画面。
---

# 图片生成与编辑

使用 `scripts/generate_image.py` 生成或编辑图片。以本文件所在目录为基准解析脚本路径。

## API 选择

- 首次选择默认使用 OpenAI Images 协议；只有用户明确指定 Gemini 时才选择 Gemini 协议。API 地址可以配置为官方服务或兼容网关。
- 延续已有 session 时读取其 `api_format`，显式传入同一 `--api-format`；缺失该字段的旧 session 按 OpenAI 处理。这是在延续已选择的 API，不是切换。CLI 每次调用仍默认 OpenAI，不会自动继承 session 的 API。
- 不根据模型名、已有密钥或失败结果推断或切换 API。
- OpenAI 和 Gemini 的地址、密钥、模型、认证方式与能力配置完全独立，不交叉兜底。
- 用户指定模型 ID 时原样传入 `--model`；否则同一 API 编辑继承 session 模型，新请求使用所选 API 的模型配置。
- 用户明确要求跨 API 编辑时，选择目标 `--api-format` 并传 `--allow-api-switch`；不要自行加该开关。
- API 失败后报告脱敏后的错误说明、错误码和 HTTP 状态。仅在任务允许时重试，不静默换模型或 API，不转贴含鉴权信息的上游原文。

首次配置或排查配置时执行 `doctor`；正式调用前可用 `--dry-run` 检查请求计划。完整配置、参数和输出字段见 [references/cli.md](references/cli.md)。

## 模式

- `generate`：根据文本生成新图片。
- `reference`：借用参考图的指定视觉特征创作新画面；构图、主体、配色等保留项和变化项由用户要求决定，不强制全新构图，不把它表述为编辑原图。
- `edit`：编辑现有图片；优先通过独立 session 延续上一张结果，没有 session 时使用 `--image`。

将用户要求与相关上下文整理成自足提示词。直接要求放入 `--prompt` 或 `--prompt-file`，辅助材料使用 `--context` 或 `--context-file`。只加入有依据的主体、构图、环境、风格、光线、色彩、保留项和排除项，不加入密钥或猜测内容。

## 执行

1. 单图直接选择 `generate`、`reference` 或 `edit`。每张需要独立迭代的图片使用独立 session。
2. 多图片或 manifest 任务先读 [references/image-generation-contract.md](references/image-generation-contract.md)，为每张图分配唯一 `asset_id`，并按依赖顺序串行执行。
3. 不覆盖用户输入或已有输出。本地保存失败且返回 `recovery_journal` 时执行 `recover --journal`；原输出路径冲突时使用 `recover --journal ... --output NEW_IMAGE` 改存已生成结果。无恢复日志时先排查保存故障，不重新调用 API。
4. `status: generated` 只表示图片已完整解码并保存。交付前查看成品，核对主体、构图、文字、瑕疵、真实比例和系列一致性。
5. 明显问题通过该图片自己的 session 修正；未经用户要求不生成付费备选。
6. 返回图片、session 和实际 manifest 的绝对路径，并说明 API、模型和尺寸。

不得把 API Key 写入提示词、session、manifest、源码或回复。
