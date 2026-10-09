# 图片生成与编辑

通用图片技能，支持 OpenAI Images 与 Gemini `generateContent` 协议，可执行文生图、参考图生图和持续图片编辑。首次默认选择 OpenAI Images 协议；用户明确指定 Gemini 时选择 Gemini 协议。API 地址可配置为官方服务或兼容网关。

## 安装

需要 Python 3.9 或更高版本：

```bash
python -m pip install -r requirements.txt
```

将 [.env.example](.env.example) 复制为根目录 `.env` 并填写密钥。示例使用同时支持这两种协议的 Skillsvc 网关；也可替换成其他兼容网关或官方地址。地址、密钥、模型 ID 和认证方式须对应同一服务商，模型别名不一定能跨服务商使用。两套配置互相独立：

| 配置 | OpenAI（默认） | Gemini（显式选择） |
|---|---|---|
| 地址 | `OPENAI_API_BASE_URL` | `GEMINI_API_BASE_URL` |
| 密钥 | `OPENAI_API_KEY` | `GEMINI_API_KEY` |
| 模型 | `OPENAI_MODEL` | `GEMINI_MODEL` |
| 认证 | `OPENAI_API_AUTH` | `GEMINI_API_AUTH` |

配置检查只读取本地配置，不发起网络请求，也不输出密钥：

```bash
python scripts/generate_image.py doctor
python scripts/generate_image.py doctor --api-format gemini
```

进程环境变量优先于 `.env`。也可使用 `--env-file` 或 `IMAGE_API_ENV_FILE` 显式选择可信文件。CLI 默认只自动读取技能根目录的 `.env`，API 地址必须使用 HTTPS，请求不跟随重定向。

## 快速开始

```bash
# 文生图，默认 OpenAI
python scripts/generate_image.py generate --prompt "一座雨后清晨的未来城市"

# 参考图生图，保留与变化项由提示词决定
python scripts/generate_image.py reference --image C:/images/reference.png --prompt "保留左右布局和配色，把汽车替换为咖啡杯"

# 继续编辑
python scripts/generate_image.py edit --session C:/images/session.json --prompt "让背景更明亮"

# 明确使用 Gemini
python scripts/generate_image.py generate --api-format gemini --aspect-ratio 3:4 --image-size 2K --prompt "简洁的产品海报"

# 延续上一次的 Gemini session，仍显式传同一 API
python scripts/generate_image.py edit --api-format gemini --session C:/images/gemini-session.json --prompt "让背景更明亮"

# 仅检查请求计划
python scripts/generate_image.py generate --prompt "测试提示词" --dry-run
```

CLI 每次调用仍默认 OpenAI，不会自动继承 session 的 API。技能延续已有图片时读取 session 的 `api_format` 并显式传入同一 API；Gemini session 省略 `--api-format gemini` 会在请求前被拒绝。确实需要把已有 session 切到另一套 API 时，必须同时明确目标 `--api-format` 和 `--allow-api-switch`。

成功输出包含图片和 session 的绝对路径、API、模型、实际参数、HTTP 状态、请求 ID、用量与耗时。参数解析与执行失败在 stderr 输出脱敏后的 JSON，包含 `error_code`、`stage`、`provider`、`retryable` 和 `http_status`；参数解析退出码为 2，其余失败为 1。保存待恢复错误还包含 `recovery_journal`，用 `recover --journal` 完成本地保存；输出冲突时可加 `--output` 改存已生成结果。`status: generated` 表示文件已通过机器校验并保存，仍需视觉检查。

完整命令、能力配置和恢复流程见 [CLI 文档](references/cli.md)；多图片 manifest 约定见 [生成契约](references/image-generation-contract.md)。

## 验证

```bash
python -X utf8 -B -m unittest discover -s tests -v
```

测试使用假密钥、临时目录、mock 和本地 HTTP Server，不调用外部图片 API。
