# 技术实验报告：税务文本结构化提取与 QLoRA 微调

所有指标来自本项目保存的真实运行记录。模型为 Qwen2.5-1.5B-Instruct，4-bit NF4＋双重量化；原始模型与微调模型采用相同加载精度及贪心解码。

## 数据与任务约定

训练集40条、验证集8条、测试集15条，另有6条独立训练流程验证样本。数据由AI助手编写并逐项核对标注、通过程序校验；尚需用户复核，不称为真实企业数据。测试集在模型实验前冻结，文件哈希见 data/manifest.json。

输出采用 action/data/reason 三个键。抽取数据保留参考字段；未知金额和税率可为null，missing_fields使用字段路径。税额只抽取明确给出的值，未知税额不用列入missing_fields。空对象、未知键、非法日期、字符串形式的数字均不能通过结构校验。

工具仅计算明确给定数值：calculate_tax_amount(amount_without_tax,tax_rate)；calculate_vat_deduction(tax_amount,deductible_ratio)。抵扣比例来自用户给定假设，不判断法规适用或抵扣资格。Decimal计算，ROUND_HALF_UP保留两位小数。参数缺失、冲突或无法在原文中明确找到时，运行时拦截。

## 环境与训练证据

首次训练：10个优化步骤，最大参数更新0.00099438，损失全部有限=True。
首次训练峰值allocated/reserved显存：2656/3036 MiB。
新进程离线重载后，固定输入的贪心输出一致：True。

正式训练：50个优化步骤，启动5轮，用时115.1秒。
正式训练峰值allocated/reserved显存：2932/4482 MiB；计算精度torch.bfloat16。

超参数：
```json
{
  "max_length": 512,
  "batch_size": 1,
  "gradient_accumulation": 4,
  "gradient_checkpointing": true,
  "learning_rate": 0.0001,
  "epochs": 5,
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": "all-linear",
  "max_grad_norm": 1.0
}
```
环境：
```json
{
  "python": "3.11.16",
  "packages": {
    "torch": "2.8.0+cu126",
    "transformers": "4.57.1",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "bitsandbytes": "0.48.1",
    "pydantic": "2.12.3"
  },
  "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "total_vram_mib": 6143.5,
  "cuda_runtime": "12.6"
}
```
按验证集答案损失最小选择第3轮检查点（验证损失0.0445），未用最终测试集选模型。
正式适配器新进程离线重载后，与已保存开发集首条输出一致：True。

训练与验证损失变化：

| 轮次 | 平均训练损失 | 验证损失 |
|---|---|---|
| 1 | 0.10879 | 0.08108 |
| 2 | 0.03465 | 0.06007 |
| 3 | 0.01504 | 0.04451 |
| 4 | 0.00514 | 0.05151 |
| 5 | 0.00106 | 0.10297 |

本次实际出现：后期训练损失继续降低，但验证损失高于较早检查点。这与过拟合的趋势一致，因此交付验证集选出的适配器，而非直接使用最后一轮；验证集较小，不能仅凭该趋势作强泛化结论。

## 测试集对照

| 指标 | Zero-shot | 固定2-shot | 微调后Zero-shot |
|---|---|---|---|
| JSON解析率 | 1/15 (6.7%) | 13/15 (86.7%) | 14/15 (93.3%) |
| 结构校验通过率 | 0/15 (0.0%) | 2/15 (13.3%) | 5/15 (33.3%) |
| 路由正确率 | 0/15 (0.0%) | 11/15 (73.3%) | 13/15 (86.7%) |
| 字段准确率（含空值） | 0/74 (0.0%) | 57/74 (77.0%) | 69/74 (93.2%) |
| 工具名及完整参数正确率 | 0/3 (0.0%) | 2/3 (66.7%) | 2/3 (66.7%) |
| 兜底动作与原因正确率 | 0/2 (0.0%) | 0/2 (0.0%) | 1/2 (50.0%) |
| 非空字段micro-F1 | 0.000 | 0.836 | 0.934 |
| 缺失字段micro-F1 | 0.000 | 0.000 | 0.333 |
| 空值处虚构字段次数 | 0 | 5 | 1 |
| 非工具请求上的工具调用尝试 | 0 | 1 | 1 |
| 推理时延中位数（秒） | 7.71 | 4.22 | 8.72 |
| 推理峰值allocated（MiB） | 1157.24 | 1173.47 | 1192.46 |

### 关键字段分项

| 字段 | Zero-shot | 固定2-shot | 微调后 |
|---|---|---|---|
| taxpayer_id | 0/10 | 7/10 | 9/10 |
| amount_without_tax | 0/11 | 10/11 | 11/11 |
| tax_rate | 0/11 | 10/11 | 11/11 |

## 实际失败案例与排查

开发阶段发现：聊天前缀末尾的换行可能与答案开头的左花括号发生BPE合并，直接按前缀词元数量切标签会出现边界不一致。修复为分别分词提示词和完成文本，并检查渲染文本连续性，保证训练和推理前缀一致。该问题来自真实数据检查错误日志。

原始模型在开发集上出现JSON外围的Markdown代码围栏；因此原始输出不能直接被JSON解析器接受。这是格式失败，不能由此断言其内部字段全部错误。固定2-shot基线用于检查通过示例提示是否已经足以改善该问题。
```json
{
  "id": "dev-01",
  "raw_output": "```json\n{\n  \"action\": \"extract\",\n  \"data\": {\n    \"invoice_date\": \"2026-09-01\",\n    \"taxpayer_id\": \"TEST000000000001\",\n    \"buyer_name\": \"甲研究室\",\n    \"items\": [\n      {\n        \"name\": \"电池\",\n        \"amount_without_tax\": 420,\n        \"tax_rate\": 0.13,\n        \"tax_amount\": 54.6\n      }\n    ],\n    \"missing_fields\": []\n  },\n  \"reason\": null\n}\n```"
}
```

验证集真实案例 `dev-01`：
```json
{
  "input": "核验记录：收票单位模拟甲研究室；开具日2026-09-01；受票方编码TEST00000000000101。采购清单仅电池，未税金额420元，税率13%，税额54.6元。",
  "expected": {
    "action": "extract",
    "data": {
      "invoice_date": "2026-09-01",
      "taxpayer_id": "TEST00000000000101",
      "buyer_name": "模拟甲研究室",
      "items": [
        {
          "name": "电池",
          "amount_without_tax": 420,
          "tax_rate": 0.13,
          "tax_amount": 54.6
        }
      ],
      "missing_fields": []
    },
    "reason": null
  },
  "raw_output": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-09-01\",\"taxpayer_id\":\"TEST00000000000101\",\"buyer_name\":\"甲研究室\",\"items\":[{\"name\":\"电池\",\"amount_without_tax\":420,\"tax_rate\":0.13,\"tax_amount\":54.6}],\"missing_fields\":[]},\"reason\":null}",
  "runtime_guard": {
    "status": "not_requested"
  }
}
```
差异定位：data.buyer_name。
这些失败保留在本轮指标中。后续改进需补充相应边界样本，并在新的独立测试集上验证，未对本轮最终测试做事后修复。

验证集真实案例 `dev-02`：
```json
{
  "input": "附言写着“待审批”。从票面抄出：受票人模拟乙商店，识别号TEST00000000000102，2026年9月2日开票；运输的税率为9%，未税金额170元。",
  "expected": {
    "action": "extract",
    "data": {
      "invoice_date": "2026-09-02",
      "taxpayer_id": "TEST00000000000102",
      "buyer_name": "模拟乙商店",
      "items": [
        {
          "name": "运输",
          "amount_without_tax": 170,
          "tax_rate": 0.09,
          "tax_amount": null
        }
      ],
      "missing_fields": []
    },
    "reason": null
  },
  "raw_output": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-09-02\",\"taxpayer_id\":\"TEST00000000000102\",\"buyer_name\":\"模拟乙商店\",\"items\":[{\"name\":\"无\",\"amount_without_tax\":170,\"tax_rate\":0.09,\"tax_amount\":null}],\"missing_fields\":[\"\"]},\"reason\":null}",
  "runtime_guard": {
    "status": "blocked",
    "reason": "invalid_output",
    "detail": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data\n  Value error, missing_fields must equal []; tax_amount is optional [type=value_error, input_value={'invoice_date': '2026-09... 'missing_fields': ['']}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error"
  }
}
```
差异定位：data.items.0.name。
结构检查给出的错误：
```text
1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]
extract.data
  Value error, missing_fields must equal []; tax_amount is optional [type=value_error, input_value={'invoice_date': '2026-09... 'missing_fields': ['']}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/value_error
```
这些失败保留在本轮指标中。后续改进需补充相应边界样本，并在新的独立测试集上验证，未对本轮最终测试做事后修复。

## 指标口径与结论边界

JSON解析与Schema校验分别统计，原始输出不去代码围栏、不做修复。格式失败仍保留在适用指标分母中。字段准确率含期望空值；micro-F1只比较非空字段的路径和值，金额和税率用Decimal归一比较。missing_fields按集合评分。工具准确率要求工具名与完整参数同时正确。

运行时校验与模型能力分别记录：程序拦截错误调用，不等于模型原始输出正确。对明确标记的数值采用保守来源检查，这不是任意自然语言的完备事实核验器。

虚构值与越界工具次数只能从成功解析的对象中计数。无法解析的输出不能判定这些细分错误是否存在；次数为0不代表所有失败样本都安全或无幻觉。整体解析、结构和字段指标仍保留失败样本。

测试集仅15条，工具类只有3条；结果适用于本批合成文本，不能推断生产可用性或真实税务准确率。训练数据中fallback例子少、写法有限；需要更多独立数据进一步验证。少量数据微调可能主要改善格式而未改善泛化，必须同时看Few-shot基线。

推理先进行一次预热，逐样本计时并同步CUDA。峰值显存为PyTorch allocated/reserved统计，不包含桌面和其他进程的所有占用。正式训练耗时包含验证、保存及结束检查，下载时间不计入。

实验选择和指标均保持真实；最终测试后不再根据测试答案调参。本报告未验证企业实际业务效果。
