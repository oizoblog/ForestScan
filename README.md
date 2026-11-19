# ForestScan

免费的离线优先蘑菇疑似目标扫描 PWA。**疑似蘑菇，不可作为食用鉴定依据。** 本仓库不附带训练数据或蘑菇模型，因此初次运行时真实检测不会报出目标。模拟演示的框始终带“模拟”标签。

## 1. 本机运行

需要 Node.js 20.19+ 或 22.12+、npm，以及训练阶段的 Python 3.11+。无需 Mac。

```powershell
npm ci
npm run test
npm run build
npm run dev
```

在电脑上打开 `http://localhost:5173`。先点击“启动摄像头”并允许权限；“模拟演示”可用于检查框和界面。普通局域网 HTTP 地址无法访问 iPhone 摄像头，手机测试请使用 GitHub Pages 的 HTTPS 地址。FPS 是画面绘制速度，并非模型推理速度。

## 2. 真实模型

数据须为 Ultralytics YOLO 检测格式：`images/train`、`images/val` 及对应 `labels/train`、`labels/val`，标注只含 `mushroom` 类。将实际数据放在 `data/`（被 Git 忽略），以 `data.example/` 为模板填写 `dataset.yaml` 和经过人工核查的 `DATA_LICENSE.json`。不要把占位符当成真实许可证。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/model_pipeline.py data/dataset.yaml --epochs 50 --imgsz 640
```

脚本默认从 YOLO11n 结构随机初始化训练；可明确传入 `--model yolo11n.pt` 使用 Ultralytics 的 AGPL 预训练权重。验证后输出 `models/mushroom/mushroom.onnx` 和 `manifest.json`。页面里一次选中两个文件。页面检查协议、类别、验证指标、SHA-256，并在 Worker 中试运行确认输出形状。输入必须为 float32 NCHW `[1,3,S,S]`，输出为未经 NMS 的单类 YOLO `[1,5,N]` 或 `[1,N,5]`，坐标是 letterbox 后的像素值。脚本输出的验证指标和数据许可是用户提供的声明，页面无法独立核实模型的实际训练数据或检测质量。请人工检查验证集，并用未参与训练的蘑菇和非蘑菇图片测试误报漏报。

推理固定使用 WASM、单线程 Worker；同一时间最多一项推理。页面显示画面 FPS，以及预处理、推理和总耗时。扫描记录只保存框、时间、模式和真实模式的模型 SHA-256，不保存照片。Service Worker 预缓存页面、Worker 和 WASM；用户选择的模型文件没有 HTTP URL，无法由 Service Worker 缓存，因此存入 IndexedDB。刷新或离线重开时重新验哈希、试运行，然后恢复真实模式。可点击“清除缓存模型”，且模型保存失败时会明确报错。Safari 可能清理站点存储，重要记录请定期导出。

### 本仓库的小样本实验

运行 `python scripts/prepare_oyster_subset.py` 可从 [Duman 等人的数据集](https://data.mendeley.com/datasets/hf55tkx489/1) 下载 24 张平菇种植场照片，按其 **CC BY 4.0** 许可保留署名、来源、逐图哈希和修改说明于 `data/oyster_experiment/DATA_LICENSE.json`。原始两个成熟度类别合并为单类 `mushroom`，20 张训练、4 张验证；图片和模型均在被 Git 忽略的目录。实验命令：

```powershell
.venv\Scripts\python.exe scripts/model_pipeline.py data/oyster_experiment/dataset.yaml --epochs 6 --imgsz 320 --batch 4 --model yolo11n.pt --output models/oyster_experiment
.venv\Scripts\python.exe scripts/verify_onnx.py models/oyster_experiment data/oyster_experiment/images/val/oyster_440.jpg
node scripts/verify_web_onnx.mjs models/oyster_experiment
```

本机实验的 mAP50 为 **0.0421**，误报很多，**不能作为可用的野外检测模型**。`models/oyster_experiment/` 只用于本机验证，没有打包进公开 PWA，也不会自动加载。若手动选择此模型，界面会显示“实验模型”、mAP50 和“未经独立核验”。该数据集只覆盖种植场平菇，与森林场景存在明显差异。

### Colab 森林蘑菇模型（本地待验收）

将 Colab 导出的 `mushroom.onnx` 放在 `models/mushroom_v1/`。该目录被 Git 忽略；页面不会默认加载它，也不会随 Pages 发布。使用 `scripts/create_manifest.py` 从 ONNX 图验证输入输出，自动计算 SHA-256 与文件大小，再填写数据集来源、许可、署名和 Colab 验证指标。当前本地 manifest 见 `models/mushroom_v1/manifest.json`；保留 `experimental: true`，指标由训练方提供，尚未在本项目独立复现。Roboflow 项目页标注 CC BY 4.0，但原始图片来源和额外署名要求尚待核查。

```powershell
.venv\Scripts\python.exe scripts/verify_onnx.py models/mushroom_v1 <本地测试图片路径>
node scripts/verify_web_onnx.mjs models/mushroom_v1
```

两个验证命令使用同一份预处理张量对照 Python ONNX Runtime 与 ONNX Runtime Web WASM。完成手机验收前，不应把训练集指标当作实际森林检测可靠性的保证。

Colab 同图原始输出已放在本地 `models/comparison/`（同样被 Git 忽略）。执行 `.venv\Scripts\python.exe scripts/compare_pytorch_onnx.py` 可复查 ONNX CPU 与 PyTorch 的逐元素误差，以及 `conf=0.25`、NMS IoU `0.45` 下的最终框。该图的最大绝对误差为 `0.0005493`、平均绝对误差为 `0.00001558`；两边均保留 1 个框，框 IoU 为 `0.9999997`。原图经 Ultralytics LetterBox 重建的张量与 Colab 输入完全相同；同图 Web WASM 与 ONNX CPU 输出最大差约 `0.000458`。这些结果证明同一预处理张量上的导出和 WASM 推理一致，不等于 iPhone 上的画面预处理或野外识别质量已验收。

## 3. GitHub Pages / iPhone Safari

第三阶段的逐项验收表见 [IPHONE_TEST.md](IPHONE_TEST.md)。本地 Git 仓库已配置 `oizoblog/ForestScan` 为远程；发布状态需以 GitHub Actions 和 Pages 的实际结果核对。模型目录被 Git 忽略，部署后仍需在 iPhone 的“文件”中手动选择 ONNX 与 manifest。

1. 将本仓库推送到公开 GitHub 仓库的 `main` 分支（GitHub Free 的 Pages 适用于公开仓库）。
2. 在仓库 Settings → Pages → Build and deployment 中将 Source 设为 **GitHub Actions**。`pages.yml` 自动测试、构建并部署，Vite `base` 自动使用仓库名。
3. 在 iPhone Safari 打开 `https://<用户名>.github.io/<仓库名>/`，点击启动，允许摄像头；确认后置画面、停止按钮、竖屏布局和黄色安全提示。
4. 切换“模拟演示”确认框带“模拟”字样；切回“真实模型”确认未加载模型时没有真实检测框或成功提示。
5. 选择自己训练出的 `.onnx` 与 `manifest.json`，检查加载状态、画框、FPS、推理耗时和错误日志。点击保存、导出诊断报告，确认 JSON 中有 `mode`、真实模式的 `modelSha256`、浏览器 UA、推理采样和自动测试结果；不含照片或精确位置。
6. 在 Safari 分享菜单选择“添加到主屏幕”。保持联网，首次打开主屏幕 PWA 并等页面和模型加载完成；保存一条记录。
7. 开启飞行模式、关闭 Wi-Fi，并从应用切换器退出 PWA。重新打开主屏幕图标，确认页面载入、显示“离线”、已保存记录仍在，且模型自动恢复并通过试运行。启动摄像头，检查 FPS、推理耗时与疑似框；模拟模式仍必须标注“模拟”。
8. 在飞行模式下点击导出，检查 JSON 和错误日志。若离线页面或模型恢复失败，先联网重开、等待完整加载，再重复测试并记录 Safari 版本、设备型号、错误日志。iOS 可能清理站点缓存和 IndexedDB，请对重要记录定期导出。

## 许可与成本

- Vite、TypeScript、vite-plugin-pwa、ONNX Runtime Web 均为免费开源依赖；具体版本及依赖许可证以 `package-lock.json` 与包内 LICENSE 为准。
- 训练脚本使用 Ultralytics，采用 AGPL-3.0 开源许可。使用或发布衍生代码/模型前应遵守其许可义务。本项目没有购买任何商业许可。
- 不附带第三方蘑菇数据集或权重。训练前必须逐项核查图片、标注、来源、再分发及模型使用许可。`DATA_LICENSE.json` 是人工记录，不是自动授权。
- GitHub Pages 对公开仓库可免费提供 HTTPS；离线推理不调用付费 API。首次加载 WASM 约 15 MB，可能耗时较长。`node scripts/check_build.mjs /仓库名/` 静态检查子路径与预缓存资源。

## 已知限制

已训练的小样本权重准确率不足，且无法从种植场结果推断野外表现；公开 PWA 仍默认不启用真实检测。这里完成了本机 Python/ONNX Runtime Web 推理和静态离线资源检查，尚未在实体 iPhone Safari 上测试飞行模式、摄像头和持续 FPS。此工具只标出疑似目标，绝不判断是否可食用。
