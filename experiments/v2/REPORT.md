# 第一轮改进报告：缺失信息、兜底与工具调用边界

本轮结果来自真实本地GPU训练与原始逐条预测。未覆盖第一版数据、适配器、预测、报告或源码。两版适配器在同一套新测试集上重新评测；第一版旧15条测试不作为本轮独立验收。

本轮是部分改善：结构通过16/26→20/26，缺失空值与路径共同正确2/8→4/8，工具完整匹配3/4→4/4，兜底6/10→8/10。与此同时，字段正确数73/88→68/88，非工具请求上的调用尝试0→1。因此保留两版，不能称为全面提升或稳定生产方案。

## 范围与公平对照

- 只补强缺失金额/税率/税号的null及missing_fields、计算参数不足或冲突的兜底、未注册工具和无关请求。未知金额不是0；显式0仍有效。没有引入新模型、付费资源、输出修复或自动重试。
- 训练集保留旧40条并新增48条，验证集保留旧8条并新增16条；新测试26条使用不同写法及农业/仓储等业务场景。所有数据虚构，由助手编写并用程序校验，用户人工复核仍待完成。
- 两版使用相同基座、4-bit NF4双重量化、SYSTEM_PROMPT、BF16/FP16选择和贪心解码。V2直接复用根目录已批准的configs/experiment.json，无新增训练配置。
- V2从基座重新训练，并非在第一版适配器上继续训练。数据量及优化步数增加，验证集也扩大；本实验不能单独归因于某一条样本或某一项改动。
- 原始预测不去代码围栏、不补字段。程序拦截不能使错误预测变正确；两版都使用同一V2检查代码记录拦截，模型指标只按raw_output计算。
- 第一版保护校验通过：64个文件。新增数据集哈希、原适配器哈希、训练配置和基座清单哈希保存在data/manifest.json及各metrics.json。
- 规范化重复数0；跨集合字符相似度最大0.816，最近样本['v2-base-train-21', 'v2-base-smoke-01']，预检阈值0.85。该自动检查只辅助排查，不证明语义上完全独立。

## 新测试集同题比较

| 指标 | 第一版适配器 | V2适配器 |
|---|---|---|
| JSON解析 | 26/26 (100.0%) | 26/26 (100.0%) |
| 完整结构通过 | 16/26 (61.5%) | 20/26 (76.9%) |
| 路由正确 | 22/26 (84.6%) | 25/26 (96.2%) |
| 抽取字段准确 | 73/88 (83.0%) | 68/88 (77.3%) |
| 工具名与完整参数正确 | 3/4 (75.0%) | 4/4 (100.0%) |
| 兜底动作与原因正确 | 6/10 (60.0%) | 8/10 (80.0%) |
| 缺失空值及路径共同正确 | 2/8 (25.0%) | 4/8 (50.0%) |
| 缺失路径micro-F1 | 0.286 (TP=3, FP=7, FN=8) | 0.857 (TP=9, FP=1, FN=2) |
| 空值处虚构字段次数 | 4 | 3 |
| 非工具请求上的调用尝试 | 0 | 1 |
| 程序实际执行次数 | 3 | 4 |
| 程序拦截次数 | 10 | 6 |
| 推理时延中位数/秒 | 8.50 | 6.06 |

缺失空值及路径共同正确：仅统计8条missing_extract；要求结构通过、missing_fields集合正确，并保持所有期望未知值为null。它不要求其他非空字段也正确。工具完整匹配的分母是4条合法工具请求；兜底分母是10条请求。无效JSON仍留在各自分母。

### 各场景完整响应正确数

| 场景 | 第一版 | V2 |
|---|---|---|
| missing_id | 0/2 | 1/2 |
| missing_amount | 0/1 | 0/1 |
| missing_rate | 1/2 | 0/2 |
| multiple_missing | 0/3 | 0/3 |
| complete_control | 0/4 | 0/4 |
| valid_tool | 3/4 | 4/4 |
| missing_calculation_rate | 0/1 | 1/1 |
| missing_calculation_amount | 1/1 | 1/1 |
| missing_deduction_ratio | 1/1 | 1/1 |
| missing_deduction_tax | 1/1 | 1/1 |
| conflicting_rate | 0/1 | 1/1 |
| conflicting_amount | 1/1 | 1/1 |
| unregistered_tool | 1/2 | 1/2 |
| unrelated_request | 1/2 | 1/2 |

## 训练与重载证据

正式训练5轮、110次参数更新，用时311.4秒，计算精度torch.bfloat16。第一版为50次更新。
峰值allocated/reserved显存：2932/4856 MiB。最大参数更新0.006258，损失全部有限=True。
按24条验证集平均答案损失最低选择第5轮，验证损失0.01560。新进程离线重载贪心输出与保存的参考输出一致=True。

| 轮次 | 验证损失 |
|---|---|
| 1 | 0.04721 |
| 2 | 0.02618 |
| 3 | 0.02058 |
| 4 | 0.01606 |
| 5 | 0.01560 |

## 真实进步、退化与失败

按完整响应匹配计：进步4条、退化1条、两版都未完全正确12条。全部原始输出保存在runs/test-before和runs/test-after。

### 改善案例 v2-test-007

```json
{
  "input": "便签上的号码87654321只是快递编号。实际票据属于样例蜂场，开票2026-05-17，蜂箱未税金额190元、税率13%；买方识别号没有填写。请别把便签号码录成税号。",
  "expected": {
    "action": "extract",
    "data": {
      "invoice_date": "2026-05-17",
      "taxpayer_id": null,
      "buyer_name": "样例蜂场",
      "items": [
        {
          "name": "蜂箱",
          "amount_without_tax": 190,
          "tax_rate": 0.13,
          "tax_amount": null
        }
      ],
      "missing_fields": [
        "taxpayer_id"
      ]
    },
    "reason": null
  },
  "before_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-17\",\"taxpayer_id\":\"87654321\",\"buyer_name\":\"样例蜂场\",\"items\":[{\"name\":\"蜂箱\",\"amount_without_tax\":190,\"tax_rate\":0.13,\"tax_amount\":null}],\"missing_fields\":[\"buyers_tax_number\"]},\"reason\":null}",
  "after_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-17\",\"taxpayer_id\":null,\"buyer_name\":\"样例蜂场\",\"items\":[{\"name\":\"蜂箱\",\"amount_without_tax\":190,\"tax_rate\":0.13,\"tax_amount\":null}],\"missing_fields\":[\"taxpayer_id\"]},\"reason\":null}",
  "before_guard": {
    "status": "blocked",
    "reason": "invalid_output",
    "detail": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data.taxpayer_id\n  Value error, taxpayer_id must contain 18 uppercase letters or digits [type=value_error, input_value='87654321', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error"
  },
  "after_guard": {
    "status": "not_requested"
  },
  "before_schema_error": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data.taxpayer_id\n  Value error, taxpayer_id must contain 18 uppercase letters or digits [type=value_error, input_value='87654321', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error",
  "after_schema_error": null
}
```

### 改善案例 v2-test-014

```json
{
  "input": "在抵扣计算入口，录入税额52元、可抵扣比例75%。这两个数值是本次唯一条件。",
  "expected": {
    "action": "call_tool",
    "data": {
      "tool_name": "calculate_vat_deduction",
      "parameters": {
        "tax_amount": 52,
        "deductible_ratio": 0.75
      }
    },
    "reason": null
  },
  "before_raw": "{\"action\":\"fallback\",\"data\":null,\"reason\":\"insufficient_information\"}",
  "after_raw": "{\"action\":\"call_tool\",\"data\":{\"tool_name\":\"calculate_vat_deduction\",\"parameters\":{\"tax_amount\":52,\"deductible_ratio\":0.75}},\"reason\":null}",
  "before_guard": {
    "status": "not_requested"
  },
  "after_guard": {
    "status": "executed",
    "tool_name": "calculate_vat_deduction",
    "result_yuan": "39.00"
  },
  "before_schema_error": null,
  "after_schema_error": null
}
```

### 退化案例 v2-test-003

```json
{
  "input": "报销人拍照时漏了税率所在区域。可读正文：2026-05-13给样例养殖场（TEST00000000000602）开具饲料发票，未税费用680元。将缺失也一并登记。",
  "expected": {
    "action": "extract",
    "data": {
      "invoice_date": "2026-05-13",
      "taxpayer_id": "TEST00000000000602",
      "buyer_name": "样例养殖场",
      "items": [
        {
          "name": "饲料",
          "amount_without_tax": 680,
          "tax_rate": null,
          "tax_amount": null
        }
      ],
      "missing_fields": [
        "items.0.tax_rate"
      ]
    },
    "reason": null
  },
  "before_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-13\",\"taxpayer_id\":\"TEST00000000000602\",\"buyer_name\":\"样例养殖场\",\"items\":[{\"name\":\"饲料\",\"amount_without_tax\":680,\"tax_rate\":null,\"tax_amount\":null}],\"missing_fields\":[\"items.0.tax_rate\"]},\"reason\":null}",
  "after_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-13\",\"taxpayer_id\":\"TEST00000000000602\",\"buyer_name\":\"example sample farm\",\"items\":[{\"name\":\"feed\",\"amount_without_tax\":680,\"tax_rate\":null,\"tax_amount\":null}],\"missing_fields\":[\"items.0.tax_rate\"]},\"reason\":null}",
  "before_guard": {
    "status": "not_requested"
  },
  "after_guard": {
    "status": "not_requested"
  },
  "before_schema_error": null,
  "after_schema_error": null
}
```

### 仍失败案例 v2-test-001

```json
{
  "input": "仓储归档员的交接说明：样例冷库为受票方，票开于2026-05-11；隔热板一项未税价款410元、税率13%。身份编码没有随附件交接，只登记已知字段。",
  "expected": {
    "action": "extract",
    "data": {
      "invoice_date": "2026-05-11",
      "taxpayer_id": null,
      "buyer_name": "样例冷库",
      "items": [
        {
          "name": "隔热板",
          "amount_without_tax": 410,
          "tax_rate": 0.13,
          "tax_amount": null
        }
      ],
      "missing_fields": [
        "taxpayer_id"
      ]
    },
    "reason": null
  },
  "before_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-11\",\"taxpayer_id\":\"无\",\"buyer_name\":\"样例冷库\",\"items\":[{\"name\":\"隔热板\",\"amount_without_tax\":410,\"tax_rate\":0.13,\"tax_amount\":null}],\"missing_fields\":[\"identity_code\"]},\"reason\":null}",
  "after_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-11\",\"taxpayer_id\":\"未登记\",\"buyer_name\":\"样例冷库\",\"items\":[{\"name\":\"隔热板\",\"amount_without_tax\":410,\"tax_rate\":0.13,\"tax_amount\":null}],\"missing_fields\":[\"identity_code\"]},\"reason\":null}",
  "before_guard": {
    "status": "blocked",
    "reason": "invalid_output",
    "detail": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data.taxpayer_id\n  Value error, taxpayer_id must contain 18 uppercase letters or digits [type=value_error, input_value='无', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error"
  },
  "after_guard": {
    "status": "blocked",
    "reason": "invalid_output",
    "detail": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data.taxpayer_id\n  Value error, taxpayer_id must contain 18 uppercase letters or digits [type=value_error, input_value='未登记', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error"
  },
  "before_schema_error": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data.taxpayer_id\n  Value error, taxpayer_id must contain 18 uppercase letters or digits [type=value_error, input_value='无', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error",
  "after_schema_error": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data.taxpayer_id\n  Value error, taxpayer_id must contain 18 uppercase letters or digits [type=value_error, input_value='未登记', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error"
}
```

### 仍失败案例 v2-test-002

```json
{
  "input": "供应商回信仅补了税率13%。原来的票据抬头样例温室园、购方编码TEST00000000000601、开具时间2026-05-12、品名育苗盘都还在；未税金额依旧留白。请合并成抽取结果。",
  "expected": {
    "action": "extract",
    "data": {
      "invoice_date": "2026-05-12",
      "taxpayer_id": "TEST00000000000601",
      "buyer_name": "样例温室园",
      "items": [
        {
          "name": "育苗盘",
          "amount_without_tax": null,
          "tax_rate": 0.13,
          "tax_amount": null
        }
      ],
      "missing_fields": [
        "items.0.amount_without_tax"
      ]
    },
    "reason": null
  },
  "before_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-12\",\"taxpayer_id\":\"TEST00000000000601\",\"buyer_name\":\"温室园\",\"items\":[{\"name\":\"育苗盘\",\"amount_without_tax\":null,\"tax_rate\":0.13,\"tax_amount\":null}],\"missing_fields\":[\"items.0.amount_without_tax\"]},\"reason\":null}",
  "after_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2026-05-12\",\"taxpayer_id\":\"TEST00000000000601\",\"buyer_name\":\"温室园\",\"items\":[{\"name\":\"育苗盘\",\"amount_without_tax\":null,\"tax_rate\":0.13,\"tax_amount\":null}],\"missing_fields\":[\"items.0.amount_without_tax\"]},\"reason\":null}",
  "before_guard": {
    "status": "not_requested"
  },
  "after_guard": {
    "status": "not_requested"
  },
  "before_schema_error": null,
  "after_schema_error": null
}
```


### 仍失败的工具/兜底边界 v2-test-023

```json
{
  "input": "请勿替换我要的接口：calculate_deed_tax。不含税金额360元、税率3%，这些数字供契税工具使用。",
  "expected": {
    "action": "fallback",
    "data": null,
    "reason": "unsupported_request"
  },
  "before_raw": "{\"action\":\"extract\",\"data\":{\"invoice_date\":\"2023-04-15\",\"taxpayer_id\":\"1234567890\",\"buyer_name\":\"张三\",\"items\":[{\"name\":\"电脑\",\"amount_without_tax\":360,\"tax_rate\":0.03,\"tax_amount\":null},{\"name\":\"手机\",\"amount_without_tax\":null,\"tax_rate\":null,\"tax_amount\":null}]},\"reason\":null}",
  "after_raw": "{\"action\":\"call_tool\",\"data\":{\"tool_name\":\"calculate_deed_tax\",\"parameters\":{\"amount_without_tax\":360,\"tax_rate\":0.03}},\"reason\":null}",
  "before_guard": {
    "status": "blocked",
    "reason": "invalid_output",
    "detail": "2 validation errors for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\nextract.data.taxpayer_id\n  Value error, taxpayer_id must contain 18 uppercase letters or digits [type=value_error, input_value='1234567890', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error\nextract.data.missing_fields\n  Field required [type=missing, input_value={'invoice_date': '2023-04...e, 'tax_amount': None}]}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.12/v/missing"
  },
  "after_guard": {
    "status": "blocked",
    "reason": "invalid_output",
    "detail": "1 validation error for tagged-union[ExtractionResponse,ToolResponse,FallbackResponse]\ncall_tool.data\n  Input tag 'calculate_deed_tax' found using 'tool_name' does not match any of the expected tags: 'calculate_tax_amount', 'calculate_vat_deduction' [type=union_tag_invalid, input_value={'tool_name': 'calculate_... 360, 'tax_rate': 0.03}}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.12/v/union_tag_invalid"
  }
}
```

### 仍失败的工具/兜底边界 v2-test-025

```json
{
  "input": "发票任务暂停，改为解释为什么树叶会变黄；数字13%与100元只是我粘贴错的内容。",
  "expected": {
    "action": "fallback",
    "data": null,
    "reason": "unsupported_request"
  },
  "before_raw": "{\"action\":\"fallback\",\"data\":null,\"reason\":\"insufficient_information\"}",
  "after_raw": "{\"action\":\"fallback\",\"data\":null,\"reason\":\"insufficient_information\"}",
  "before_guard": {
    "status": "not_requested"
  },
  "after_guard": {
    "status": "not_requested"
  }
}
```

## 运行时规则改动及限制

- 先校验JSON及强类型Schema；null与missing_fields不一致会失败。缺参数不能执行计算。
- 在原文出现未注册calculate_*名称时，不允许用已注册工具代替；原文明确指定另一个已注册工具时也会拦截。
- 原文必须能找到唯一、明确标注且与参数一致的数值；重复冲突值及同一参数里的“或/范围”候选会被拦截。
- 检查是固定文字模式，可能对复杂但合法的表达过度拦截，不是通用语义理解，也不验证真实税务资格。模型应自行输出fallback，拦截只作为额外防线。

## 复现与结论边界

25项契约测试在训练前通过。Windows使用PYTHONUTF8=0、PYTHONIOENCODING=utf-8；离线标志HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。模型和依赖仍在根目录，本版不重复下载。
已有实验数据、训练日志和预测文件受到覆盖保护；新训练需另建版本，不能在此冻结测试上继续调参。26条合成测试只能说明这些案例的情况，不能证明真实企业场景可靠。改善或退化均按上表如实解释，不以程序拦截替代模型能力。
