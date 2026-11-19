# RPBH 编排层

FastAPI 服务提供角色对话、持久输入队列、角色记忆、批次历史归档、腾讯 ASR、本地 TTS、Live2D / VRM 表现和主动互动。角色 Persona 从根配置指定文件读取，独立于框架 system prompt。DSH 仍使用模拟执行器，真实插件留待后续联调。

## 启动

按[项目快速开始](../README.md#快速开始)安装依赖，并将 `config.example.json` 复制为 `config.json`。在项目根目录运行 `python -m backend`，访问 `http://127.0.0.1:8770/docs` 调试接口。服务默认监听本机。若本地已有 Web 构建产物，可通过 `server.frontend_directory` 指定静态目录。

模型凭据可由环境变量或本地文件提供。`GET /api/rp/config` 对客户端提供运行策略。SQLite、回调令牌和临时音频存放在本地配置指定的路径。

## 上下文、归档与日志

下表中的文本预算以 UTF-8 字节数作为保守 token 上界，不等同于服务端实际计费 token。

| 配置 | 默认值 | 行为 |
| --- | --- | --- |
| context.max_input_tokens | 32768 | 单次请求总上限，包含系统、人格、工具、背景、原文、当前输入与图片预留 |
| context.current_input_max_tokens | 6000 | 当前用户输入上限 |
| memory.recent_max_tokens | 6000 | 近期原文预算 |
| memory.recent_max_age_seconds | 259200 | 近期原文最多 3 天，和预算取较短范围 |
| memory.archive_summary_max_tokens | 2400 | 反复合并后的历史对话摘要上限 |
| memory.character_max_tokens | 1200 | 角色与用户长期记忆预算 |
| memory.task_snapshot_max_tokens | 1200 | 任务简报预算 |
| memory.archive_trigger_multiplier | 16 | 本地文本累积到近期预算的若干倍后触发语义归档 |
| memory.archive_input_max_tokens | 131072 | 单次归档输入容量 |
| retention.logs | 6000 / 3600 秒 | 模型响应、工具决定和任务事件日志的长度与时间上限 |
| retention.maintenance_interval_seconds | 30 | 后台维护间隔 |

默认 memory.policy 为 batched_archive。日常每轮不会反复调用摘要模型。本地文本达到阈值后，一次 API 请求合并历史摘要并提取用户明确说出的角色记忆；提交成功才删除已归档的旧原文。图像观察不能冒充用户原话作为画像证据。

归档后物理原文仅保留近期预算与 3 天约束下的完整轮次尾部；另有各自限长的摘要、角色事实及请求/任务 ID 和哈希墓碑。模型失败或归档过程中源文本、真实交付回执变化时保留原文。遗留超容量记录按完整连续轮次分批，不截断中间内容冒充完整归档。触发倍数和各项预留超过归档容量时配置校验会拒绝启动。

近期上下文与本地攒批原文是独立限制。总请求预算不足时先去掉较老历史，再减少可选背景；固定人格和当前输入仍无法容纳时明确报错，不自动扩张上下文。记忆覆盖过的原文不会重复注入。

会话日志超过长度或 1 小时任一边界即清理正文，保留幂等哈希和任务最新状态。维护会 checkpoint；新数据库使用 incremental vacuum，旧数据库复用释放的页面。

## 会话和消息 API

- POST /api/rp/conversations 创建或恢复活动会话。
- POST /api/rp/messages 接收 conversation_id、request_id、text、source 及可选 images，202 表示已经持久入队。旧单数 /api/rp/message 可等待本轮结果。
- GET /api/rp/conversations/{id} 返回文字记录、待交付表现、任务、音频状态和记忆。
- POST /api/rp/conversations/{id}/deliveries/{action_id} 确认真正展示的全文或 text 前缀。
- PATCH /api/rp/conversations/{id}/presentation 切换形象；工具枚举同步切换，旧版本模型结果提交前会重新校验。
- POST /api/rp/conversations/{id}/archive 以 202 返回 archiving；停止旧表现、处理已有输入、强制归档后标记 archived。失败保留原文，可重试；中途重启会恢复。
- POST /api/rp/conversations/{id}/activity 接收 typing、speech、camera、image 或 interaction，表示外界输入。

前端菜单提供“归档并开始新对话”，先等待已展示文字回执，再提交归档并切换新会话。“开始新对话”单独创建会话，旧会话数据仍在。归档后的旧 request_id 不会再次派任务。普通会话默认在刷新时恢复，挂断不会清空已确认记录。

/ws/rp/{id} 推送 input.accepted、turn.started/completed、rp.message、audio.updated、rp.interrupted、presentation.changed、conversation.archived 等事件。resync_required 要求客户端重新获取快照；断线不会撤销已持久入队的消息。大回放同时监督发送和接收，断线可取消回放，不会堵死队列。

## 音频只作临时播放缓存

TTS 保留旧项目分句、语音清理和 GPT-SoVITS 参数。音频临时写入 data/audio/<action_id>.wav，文件响应禁止 HTTP 缓存。

- 播放完成即删文件；默认中断和失败也清理。
- 无播放回执最多保留 1 小时，总缓存上限 64 MiB。
- playing 漏报结束时，音频时长加 30 秒宽限后清理。
- Windows 文件句柄暂时占用时稍后重试。
- 只清 data/audio，不处理 GPT-SoVITS 自己的模型或推理缓存。

POST /api/rp/conversations/{id}/playback/{action_id} 独立记录 playing/completed/interrupted/failed。生成完成不代表已经交付。轻量 job 保留 file_state=deleted 防止刷新、重复回执或重启重新合成；已删文件 GET /api/rp/audio/{action_id}/file 返回 410。文字交付历史独立保留。

## 图片与视频

图片元素为 image_id、data_url、captured_at、source。每轮默认最多 3 张。原图只在有数量、字节数和 TTL 限制的内存中存在，不写数据库或日志，也不会进入后续轮次。后端以受并发限制的线程校验、缩放和转码。

同一次 Responses 请求返回 RP、可选任务决定，以及逐图 image_summarize 摘要；没有额外串行识图请求。摘要替换会话中的图片占位内容后释放像素。失败、取消和修复路径同样清理；重启后缺失原图的未完成轮次会失败并要求重发，不能无图继续派发任务。

| 图片策略 | 默认 |
| --- | --- |
| 摄像头 | 长边 960，WebP 质量 75，上限 192 KiB，detail=low |
| 用户上传 | 长边 1600，质量 85，上限 512 KiB，detail=original |
| 前端编码等待 | 400ms；超过即丢帧，不阻塞 PCM/VAD |
| 后端处理 | 最多 2 个任务、3 秒期限、12 次编码尝试 |
| 临时图片缓存 | 16 MiB、32 张、120 秒，活动请求中的图不会被逐出 |
| 周期采样 | 每 4 秒，两帧一批，分析至少间隔 8 秒 |

周期图忙时丢弃，不积压；真实用户输入优先取消未完成的自动视觉观察。自动观察只开放 RP 和图片摘要，不开放任务工具。用户说话结束时立即复制摄像头像素，另走语音轮次，不等待周期批次。

/ws/asr/{id} 首包为 start，continuous=true 表示连续通话，video=true 表示视频。silence 立即携带 capture_id；异步编码完成再发 camera.frame。后端绑定当前说话阶段，续话或提交后的迟到帧丢弃。单次录音 end 可带 images，成功入队后前端才清对应草稿。连续通话的明确 restart 可重连；网络错误会明确结束。

## 无输入互动

IdleScheduler 独立于模型运行。默认每 5 秒检查当前连接中的活动会话，300 秒没有外界输入时，将 idle.prompt 作为内部事件提交到 loop，仅开放 rp_perform。一次安静期间默认只提醒一次。

打字、实际说话、图片、摄像头与任务回调重置计时；轮询、模型回复和提醒本身不重置。项目退出后不运行；归档、未连接的历史会话不会群发。等待阈值、检查频率、是否重复及提示词都在根配置中。

## 模拟任务回调和验证

POST /api/internal/task-events 使用配置文件指定的本地 Bearer 令牌，支持 progress/completed/failed/cancelled 与可选 result_ref。事件哈希防重放，终态拒绝迟到进度；任务播报只开放 RP，不再次派发原任务。DSH 的真实执行插件仍待后续联调。

可以通过 `/docs` 验证会话接口。任务回调的 Bearer 令牌取自 `server.callback_token_file`，默认保存在 `data/state/task-callback.token`。

Android 配对与传输由 `backend/mobile/` 提供，参数见根配置的 `mobile` 段。HTTPS 信令地址、房间、令牌和 ICE 服务使用 [本地连接配置模板](../android/config/)。
