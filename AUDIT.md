# ForestScan 审查记录（2026-11-18）

## 问题与处理

| 严重程度 | 问题 | 处理 |
| --- | --- | --- |
| 高 | 模型推理异常可能使主线程停留在 busy 状态 | Worker 请求加编号和 60 秒超时，失败时关闭 Worker 并清除检测框 |
| 高 | 用户选取的 ONNX 无法由 Service Worker 缓存，刷新后丢失 | ONNX 存 IndexedDB；恢复时重新验 SHA-256 并试运行。静态资源与 WASM 仍由 Service Worker 预缓存 |
| 中 | 输出数据长度、形状及大量候选框未严格约束 | 加形状与长度校验，NMS 先取最高 300 候选并最多保留 50 框 |
| 中 | manifest 自述数据集和指标不可独立核验 | UI 显示“用户/实验模型”“未经独立核验”及 mAP50；记录保存模型 SHA-256 |
| 中 | Safari 权限、播放失败或后台恢复可能留下摄像头轨道 | 权限失败时停止轨道，返回后台时停止摄像头，由用户再次点击启动 |
| 低 | 无推理耗时与错误日志 | UI 显示预处理、推理与总耗时；错误保存于 IndexedDB 并随记录导出 |

## 本地验证

- `npm run test`：4 项几何、NMS 与诊断报告隐私测试通过。
- GitHub Pages 子路径生产构建：通过；`scripts/check_build.mjs /ForestScan/` 检查 11 个预缓存项，包含 Worker、WASM、图标和 manifest。
- 本机预览 `/ForestScan/`、manifest、Service Worker 请求均返回 HTTP 200，浏览器页面呈现正常。
- 24 张平菇种植场照片及框标注来自 Duman 等人《Annotated Oyster Mushroom Images》，CC BY 4.0；20 训练、4 验证，类别合并为 mushroom。来源、署名、改动及逐图哈希在 `data/oyster_experiment/DATA_LICENSE.json`。
- YOLO11n 预训练权重、CPU、6 epochs、320px：验证 mAP50 **0.0421**，mAP50-95 **0.0158**。实验模型在 `models/oyster_experiment/`，未公开部署。
- 实际验证图 ONNX 输出 `[1,5,2100]`；Python ONNX Runtime 与 ONNX Runtime Web WASM 的最大元素差 **0.00110**。浏览器文件选择器加载模型的尝试中，自动化连接中断，故未验证页面内的模型恢复。

## 剩余风险

实验模型误报很多，而且只见过种植场平菇；不可用于森林识别或食用判断。manifest 来源和指标无法由前端独立证明。实体 iPhone Safari 摄像头、长时间 FPS、飞行模式以及 IndexedDB 被系统清理后的恢复尚未实测。本地 Git 仓库已初始化为 main，但没有远程地址，因此 GitHub Pages 工作流已配置但未发布。第三阶段逐项验收见 `IPHONE_TEST.md`。

## 第三阶段部署检查

- `scripts/check_build.mjs /ForestScan/` 直接检查编译后注册代码中的 Service Worker URL/scope、Worker URL、WASM URL、manifest scope/start URL，以及预缓存文件存在性。
- 诊断 JSON 包含浏览器 UA、模型 SHA-256、推理耗时/FPS、错误、自动测试结果和扫描记录；不含照片、视频帧、模型字节或精确位置。人工验收结果记录在 `IPHONE_TEST.md`。
- `.gitignore` 排除训练数据、实验模型、ONNX/预训练权重、虚拟环境、构建产物和常见密钥文件。GitHub 授权连接为 `oizoblog`，授权仓库中未发现 `ForestScan`；没有创建公开仓库或推送。
