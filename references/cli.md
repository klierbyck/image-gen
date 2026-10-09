# CLI 调用约定

从技能根目录执行 `scripts/generate_image.py`。依赖为 `requests`、`Pillow` 和 `filelock`。

## 配置与检查

CLI 默认只自动加载技能根目录 `.env`；进程环境变量优先。`--env-file PATH` 或 `IMAGE_API_ENV_FILE` 可选择其他可信配置文件。不要通过命令行传递密钥。

OpenAI 配置：

- `OPENAI_API_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`
- `OPENAI_API_AUTH=bearer|x-goog-api-key`
- `OPENAI_DEFAULT_ASPECT_RATIO`、`OPENAI_DEFAULT_IMAGE_SIZE`、`OPENAI_DEFAULT_SIZE`
- `OPENAI_DEFAULT_OUTPUT_FORMAT`、`OPENAI_DEFAULT_QUALITY`
- `OPENAI_SUPPORTED_FORMATS`、`OPENAI_SUPPORTED_QUALITIES`、`OPENAI_MAX_INPUT_IMAGE_MB`

Gemini 使用对应的 `GEMINI_` 前缀配置。两套配置不交叉兜底。每次 CLI 调用默认 OpenAI；只有 `--api-format gemini` 才使用 Gemini。模型名、session 和密钥是否存在都不会改变 CLI 的选择。延续 Gemini session 的技能调用须读取已有 `api_format` 并显式传入 `--api-format gemini`。

内置模型默认值为 OpenAI `gpt-image-2`、Gemini `nana-banana-2`。默认比例 `16:9`、尺寸档位 `2K`、输出格式 `png`、质量 `auto`、输入图上限 4 MB。`*_DEFAULT_OUTPUT_FORMAT` 和 `*_DEFAULT_QUALITY` 可设置有效默认值，必须分别属于 `*_SUPPORTED_FORMATS` 和 `*_SUPPORTED_QUALITIES`。`*_DEFAULT_SIZE` 是 OpenAI Images 的可选精确像素默认值；设置后在未传尺寸参数的新请求中优先使用。

支持格式和质量使用逗号分隔。OpenAI 内置格式为 `png,jpeg,webp`，质量为 `auto,low,medium,high`；Gemini 当前适配器只输出 PNG，默认格式必须为 `png`，默认质量必须为 `auto`，支持集合也必须包含这两个值。若 OpenAI 服务只支持 JPEG 和 high，须同时设置 `OPENAI_SUPPORTED_FORMATS=jpeg`、`OPENAI_DEFAULT_OUTPUT_FORMAT=jpeg`、`OPENAI_SUPPORTED_QUALITIES=high`、`OPENAI_DEFAULT_QUALITY=high`。

地址可为根地址或包含 `/v1`、`/v1beta` 的地址，必须是无用户名、密码、查询参数和片段的 HTTPS URL。`--base-url` 只覆盖所选 API 的地址，不改变其密钥或模型。

`openai`、`gemini` 表示请求协议，服务商由地址决定。`.env.example` 保留 Skillsvc 兼容网关示例；官方地址可分别使用 `https://api.openai.com` 和 `https://generativelanguage.googleapis.com`，并配置对应服务的密钥、实际模型 ID、鉴权方式和能力参数。使用 Gemini 官方 API 时配置 `GEMINI_API_AUTH=x-goog-api-key`。进程环境变量优先，不会因更改地址而自动更换已有密钥。

检查配置，不联网且不打印密钥：

```bash
python scripts/generate_image.py doctor
python scripts/generate_image.py doctor --api-format gemini --env-file C:/config/image.env
```

`doctor` 检查所选 API 的地址、密钥、模型、认证方式、能力配置及默认参数与支持集合的一致性。缺少密钥或配置无效时返回结构化错误。它不联网，因此不能证明模型存在、额度充足或服务端实际接受所有参数。

## 命令

```bash
python scripts/generate_image.py generate --prompt "完整图片要求"

python scripts/generate_image.py reference \
  --image C:/images/style.png \
  --prompt "保留参考图左右布局和配色，把汽车替换为咖啡杯"

python scripts/generate_image.py edit \
  --session C:/images/session.json \
  --prompt "缩小标签，保持构图和配色不变"
```

模式：

- `generate`：纯文本生成。
- `reference`：`--image PATH` 可重复，或用 `--reference-asset-id` 引用 manifest 资产。构图等保留项和变化项由提示词决定。
- `edit`：用 `--image` 开始编辑，或用 `--session` 延续上一张输出。
- `doctor`：检查本地配置。
- `recover`：从 `.pending.json` 完成本地保存，不调用 API。

通用参数：

- `--api-format openai|gemini`：默认 OpenAI。
- `--model MODEL_ID`：覆盖所选 API 的模型，不切换 API。
- `--base-url URL`：覆盖所选 API 的地址。
- `--prompt TEXT` 或 `--prompt-file PATH`：二选一。
- `--context TEXT`、`--context-file PATH`：均可重复。
- `--aspect-ratio W:H`、`--image-size 1K|2K|4K`。
- `--size WIDTHxHEIGHT`：OpenAI Images 精确尺寸，宽高至少 256 且为 16 的倍数，比例不超过 3:1；不能与比例或尺寸档位同时使用。
- `--quality auto|low|medium|high`、`--output-format png|jpeg|webp`、`--compression 0..100`：OpenAI Images 参数。
- `--output-dir DIR`、`--output PATH`、`--session PATH`。
- `--manifest PATH --asset-id ID`：必须成对使用。
- `--env-file PATH`、`--timeout SECONDS`、`--dry-run`。
- `edit --allow-api-switch`：仅在用户明确要求时允许 session 跨 API 编辑。

Gemini 不接受 `--size`、`--quality`、`--compression` 或非 PNG 输出。OpenAI Images 接收精确像素 `size`；Gemini 接收 `aspectRatio` 和 `imageSize`。

## Session 与跨 API

同一 API 编辑时，模型和未覆盖的生成参数可从 session 继承。旧 session 没有 `api_format` 时按 OpenAI 处理。

CLI 不会自动继承 API。已有 Gemini session 继续编辑须显式传同一 API，这是延续已有选择，无需 `--allow-api-switch`：

```bash
python scripts/generate_image.py edit \
  --session C:/images/gemini-session.json \
  --api-format gemini \
  --prompt "让背景更明亮"
```

若本次 API 与 session 不同，CLI 默认在请求前拒绝。确认切换时显式执行：

```bash
python scripts/generate_image.py edit \
  --session C:/images/session.json \
  --api-format gemini \
  --allow-api-switch \
  --prompt "使用 Gemini 继续编辑"
```

切换后使用目标 API 的地址、密钥、模型和能力默认值，不继承另一 API 的参数。

## 输出与错误

成功时标准输出为一个 JSON 对象。主要字段：

- `image`、`session`、`manifest`：绝对路径。
- `api_format`、`model`、`endpoint`、`parameters`。
- `http_status`、`request_id`、`usage`、`elapsed_seconds`。
- `actual_dimensions`、`mime_type`、`bytes`、`status`。

request 元数据同时写入 session turn 和 manifest 资产记录。`status: generated` 只表示完整解码和本地保存成功，不代表视觉质量通过。

参数解析与执行失败时，标准错误为一个脱敏后的 JSON 对象；已配置密钥和鉴权字段会被掩码。成功退出码为 0，参数解析失败为 2（`error_code: argument_error`、`stage: arguments`），其他失败为 1：

```json
{
  "error": "错误说明",
  "error_code": "validation_error",
  "stage": "validation",
  "provider": "openai",
  "retryable": false,
  "http_status": null
}
```

`retryable` 只是机器可读提示；超时不证明服务端未生成结果，不能盲目重复付费调用。

图片下载失败使用 `stage: download` 并保留 HTTP 状态及可重试信息；应优先排查或重试已有图片下载，不能直接重跑付费生成。保存已有结果失败时使用 `stage: storage`：有恢复日志的错误为 `error_code: storage_pending`，并包含绝对路径 `recovery_journal`；无法建立恢复日志时为 `storage_failed`。这两种错误都不应重新调用生成 API。

## 保存、并发与恢复

默认输出目录为 `generated_images/`。输出不会覆盖已有文件，扩展名按实际图片格式修正。输入图片执行完整解码、像素与配置上限检查。

同一 manifest 和 session 使用跨进程锁。显式 `--output` 的候选输出路径也在请求前加锁并重新检查，避免独立 session 同时为同一路径调用 API。同一批资产按依赖顺序串行执行；不同 manifest 且 session 独立时可并行。session 和 manifest 优先记录相对路径，可随整个输出目录一起移动。

保存中断会生成 `.pending.json` 并阻止继续调用 API。按错误中的绝对路径恢复：

```bash
python scripts/generate_image.py recover --journal C:/images/session.json.pending.json
```

恢复只完成本地提交，不读取密钥、不联网、不追加编辑轮次。

若旧输出位置已被不同内容占用，可以为已有 pending 结果指定一个未占用的新输出路径：

```bash
python scripts/generate_image.py recover --journal C:/images/session.json.pending.json --output C:/images/recovered.png
```

`recover --output` 保存 journal 中已有图片，同步修正 session、manifest 和返回结果中的图片路径，保留其他资产与编辑轮次。相对输出路径以 journal 所在目录为基准，扩展名按实际图片格式修正。新图片路径不能指向输入或元数据文件，不会覆盖已有图片，也不会重新生成图片。

## 协议映射

- OpenAI 文生图：`POST /v1/images/generations`，JSON。
- OpenAI 参考图或编辑：`POST /v1/images/edits`，multipart。
- Gemini 所有模式：`POST /v1beta/models/{model}:generateContent`，图片使用 `inlineData`。

其他协议需要新增适配器，不能只修改地址后假定兼容。
