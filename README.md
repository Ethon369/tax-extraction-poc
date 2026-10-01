# 税务文本结构化提取与 QLoRA 微调 PoC

本项目将文字单据转换为结构化 JSON，生成两个演示计算工具的调用，或返回明确兜底。效果以 REPORT.md 和 runs 下的实际记录为准。

## 实验摘要与交付内容

- 本机已完成 RTX 3060 Laptop 6GB 上的 QLoRA 训练、三组对照评测和新进程离线适配器重载验证。
- 模型为 Qwen2.5-1.5B-Instruct；40条训练、8条验证、15条最终测试，另有6条流程验证数据。所有数据均为合成数据。
- 相比固定2-shot，微调后字段准确率从57/74（77.0%）变为69/74（93.2%）；完整结构通过率为5/15（33.3%），工具名与完整参数正确率仍为2/3。结果不代表生产可靠性。
- 技术报告见 [REPORT.md](REPORT.md)，学习记录见 [LEARNING.md](LEARNING.md)，开发问题记录见 [DEVELOPMENT.md](DEVELOPMENT.md)。原始预测、训练日志和环境信息保存在 `runs/`。
- 仓库包含正式适配器 `runs/formal/adapter` 和流程验证适配器 `runs/smoke/adapter`。不包含完整基座模型、Python环境或下载缓存；请按下文安装依赖并下载固定版本的模型。

获取仓库：

```powershell
git -c core.autocrlf=false clone https://github.com/Ethon369/tax-extraction-poc.git
cd tax-extraction-poc
```

克隆时临时关闭自动换行转换，以保持冻结数据、配置和模型清单的原始字节及SHA256校验值；此命令不更改全局Git配置。

## 环境

- Windows x86-64，NVIDIA RTX 3060 Laptop 6GB；Python 3.11 独立环境。
- 模型 Qwen2.5-1.5B-Instruct；NF4 4-bit QLoRA，全部实验使用相同量化方式。
- 网络用于首次安装和下载；训练、推理与评测均使用本地文件。
- 预留约20GB以上磁盘空间；PyTorch的安装缓存和运行库也会占用空间。

项目已安装环境时，在项目根目录执行：

```powershell
$env:PYTHONUTF8='0'
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\check_data.py --tokenizer
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

新机器可以使用 uv，在项目根目录安装：

```powershell
$env:UV_CACHE_DIR="$PWD\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR="$PWD\.python"
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_model.py
```

若要复现本次实际安装的全部依赖版本，使用同一CUDA索引及完整锁定清单：

```powershell
uv pip install --python .venv\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.lock.txt
```

下载来源为 ModelScope 的 Qwen 仓库。每个文件按下载时固定的版本号请求并与服务端SHA256核对。`models/Qwen2.5-1.5B-Instruct/download_manifest.json`保存版本和校验值。复现包保留这个清单，因此下载脚本会复用相同版本而非重新选取最新文件。

## 分阶段运行

```powershell
# 第一阶段：GPU训练10步、适配器保存、新进程离线加载
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Stage smoke

# 后续阶段：开发集基线、正式训练、开发集检查、冻结测试集三组评测、报告
powershell -ExecutionPolicy Bypass -File scripts\run_experiment.ps1 -Stage full
```

正式评测已有结果时会拒绝覆盖，避免无意反复使用最终测试集。需要新实验时，应先归档 runs，注明原因并使用新的独立测试集。第一次训练适配器在 `runs/smoke/adapter`，正式适配器在 `runs/formal/adapter`，两者用途不同。

本机默认训练已成功，无需低显存备用方案。若在更受限设备上验证流程，可运行 `-m src.train --profile smoke --reduced`：只缩短smoke任务提示，保留全部输入和答案，并使用256词元及q_proj/v_proj。该模式不用于正式效果对比；正式数据不能自动截断到256词元。

也可以逐步运行：

```powershell
.\.venv\Scripts\python.exe -m src.train --profile smoke
.\.venv\Scripts\python.exe -m src.inference --verify-smoke
.\.venv\Scripts\python.exe -m src.evaluate --split dev --mode zero
.\.venv\Scripts\python.exe -m src.evaluate --split dev --mode few
.\.venv\Scripts\python.exe -m src.train --profile formal
.\.venv\Scripts\python.exe -m src.evaluate --split dev --mode finetuned
.\.venv\Scripts\python.exe -m src.inference --verify-formal
.\.venv\Scripts\python.exe -m src.evaluate --split test --mode zero
.\.venv\Scripts\python.exe -m src.evaluate --split test --mode few
.\.venv\Scripts\python.exe -m src.evaluate --split test --mode finetuned
.\.venv\Scripts\python.exe scripts\build_report.py
```

单条离线推理：

```powershell
$env:PYTHONUTF8='0'
$env:PYTHONIOENCODING='utf-8'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -m src.inference --adapter runs\formal\adapter --text '不含税金额100元，税率13%，帮我算税额。'
```

Windows中文环境中，某些依赖启动的系统命令可能输出本地编码；全局启用`PYTHONUTF8=1`时可能出现`UnicodeDecodeError`。本机已验证使用`PYTHONUTF8=0`并设置`PYTHONIOENCODING=utf-8`可完成此条推理且不出现该错误。项目JSON文件仍显式使用UTF-8读写。`torch_dtype is deprecated`是当前依赖的参数名提醒，不表示本次推理失败。

`adapter_config.json`保留训练时的本机基座路径，作为原始训练产物。请通过本项目的`src.inference`入口加载：它根据仓库所在位置加载本地基座，再载入适配器，不依赖训练电脑的D盘路径。

仓库已包含本轮实验记录。请勿直接运行`-Stage full`来覆盖它们：该流程在已存在评测结果时会拒绝继续。仅复现已有适配器可运行上面的单条推理，或运行`--verify-formal`；若要开展新训练，请在另一个目录归档现有`runs`后创建新的实验记录，并按独立测试集规则验证。

## 数据与接口

`data/train.json`40条、`dev.json`8条、`test.json`15条；另有6条独立smoke样本。所有公司、税号和业务记录均为虚构，TEST开头的18位编号不是实际企业统一社会信用代码。数据由助手编写并校验，尚需用户复核。数据生成脚本保留逐条输入与标准答案；现有数据已冻结，完成实验后请勿直接重建覆盖。

数据采用system/user/assistant聊天格式，另带id、category、template_family和synthetic审计字段。训练只对assistant答案计算损失，超过序列上限直接报错，不截断答案。提示词和完成文本独立分词以避免BPE在边界处合并。

响应含action/data/reason，三种动作：extract、call_tool、fallback。抽取保留完整键，未知值null，missing_fields记录关键缺失字段路径；tax_amount未知可为空且不列入missing_fields。日期校验实际有效性，税率和比例为0到1的数字。结构验证拒绝未知键、数字字符串、布尔数值、NaN和重复JSON键。

工具只接受注册名字和完整参数；参数必须在原文里以明确标签给出，并且不能冲突。calculate_vat_deduction只根据明确的tax_amount和deductible_ratio计算，未实现税法资格判断。参数语义或数值来源检查不通过会阻止执行，但不会把错误原始模型输出算作正确。

## 评测与复现证据

- Zero-shot、固定2-shot、微调后Zero-shot采用同一模型和4-bit加载，贪心解码；Few-shot例子为train-01和train-21。
- 验证集用来计算答案损失并选择检查点；最终测试冻结，最终测试后不调参。
- JSON解析率与Schema通过率分开；不修复原始输出。失败样本保留分母。
- predictions.jsonl保存每条输入、原始输出、答案、耗时和运行时拦截；metrics.json保存指标与环境、数据、模型指纹。
- 峰值显存同时记录allocated和reserved；推理预热排除在时延统计之外。
- 小数据指标不代表真实企业数据或生产可靠性。

修改训练配置前先审阅完整内容。付费API或云算力不在默认执行流程中。
