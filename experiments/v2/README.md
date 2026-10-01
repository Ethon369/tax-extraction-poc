# 第一轮改进实验（V2）

本目录独立保留缺失信息、兜底与工具调用边界改进。第一版仍位于仓库根目录，所有原数据、适配器、逐条预测和报告不被覆盖。真实对比见本目录的REPORT.md。

## 数据与实验

- 训练88条：第一版40条＋48条新增样本；验证24条：第一版8条＋16条新增样本；新测试26条，未复用第一版最终测试。另保留6条独立流程验证样本，不参与正式训练。
- 缺失金额、税率和税号：extract保留null，并准确列出missing_fields。计算所需参数缺失或冲突：fallback/insufficient_information。未注册工具和无关问题：fallback/unsupported_request。显式零值不能当成缺失。
- 从基座重新训练，直接使用根目录已批准的configs/experiment.json、同一虚拟环境和基座下载。SYSTEM_PROMPT与第一版相同。
- 同一新测试集比较第一版适配器与本版适配器；两版均使用V2的检查代码，指标按原始模型输出计算，拦截结果不算预测正确。
- 所有数据为助手编写、程序校验的合成案例；用户人工复核仍待完成。样本量小，不能推断生产表现。

## 在项目根目录运行

```powershell
$env:PYTHONUTF8='0'
$env:PYTHONIOENCODING='utf-8'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'

# 检查已有冻结数据和第一版保护哈希
Push-Location experiments\v2
& ..\..\.venv\Scripts\python.exe scripts\check_data.py --tokenizer
& ..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 使用已交付适配器做单条推理；所有模型文件均在项目根目录
& ..\..\.venv\Scripts\python.exe -m src.inference --adapter runs\formal\adapter --text '计算税额，不含税金额100元，税率未提供。'
Pop-Location
```

全新实验按prepare、train、compare三个阶段运行。已交付目录含冻结数据与结果，会拒绝重建或覆盖；以下仅用于尚无数据和结果的新版本，并需保留第一版数据以生成保护清单：

```powershell
.\.venv\Scripts\python.exe experiments\v2\scripts\run_experiment.py --stage prepare
.\.venv\Scripts\python.exe experiments\v2\scripts\run_experiment.py --stage train
.\.venv\Scripts\python.exe experiments\v2\scripts\run_experiment.py --stage compare
```

若要重新训练交付的数据，在另一个检验目录归档本版runs后运行train、compare；同一冻结测试只能作为复现检验，不可用于调参。根目录data/runs及原REPORT.md必须保留，v1_preservation.json会逐项检查。

## 文件

- data/manifest.json：数据冻结哈希及版本说明。
- v1_preservation.json：第一版64个关键文件的保护哈希。
- runs/formal/adapter：根据24条验证集选出的适配器。
- runs/formal/summary.json、training_log.json、selected_checkpoint.json：真实训练证据。
- runs/formal/reload_expected.json、reload_verified.json：新进程离线重载证明。
- runs/test-before、runs/test-after：同一新测试集的逐条原始预测、程序拦截和指标。
- tests/test_boundaries.py：缺参数、候选冲突、未注册工具替换、零值及指标口径的检查。

原文数值检查是保守的固定文字模式，会有局限；它不做税法资格推断，不修复模型答案，也不自动重试。
