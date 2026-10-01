from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, PROJECT_ROOT, load_config, read_json, read_split, sha256
from src.metrics import equal_nested, flatten
from src.schemas import strict_json_loads, validate_response
from check_data import assert_preserved

def fraction(value):
    return f'{value["correct"]}/{value["total"]} ({100*value["rate"]:.1f}%)' if value['rate'] is not None else 'N/A'

def f1(value):
    return f'{value["f1"]:.3f} (TP={value["tp"]}, FP={value["fp"]}, FN={value["fn"]})' if value['f1'] is not None else 'N/A'

def parsed(row):
    try:
        value=strict_json_loads(row['raw_output']);validate_response(row['raw_output'])
        return value, None
    except (ValueError,TypeError) as error:
        return None,str(error)

def main():
    before=read_json(ROOT/'runs/test-before/metrics.json')
    after=read_json(ROOT/'runs/test-after/metrics.json')
    train=read_json(ROOT/'runs/formal/summary.json')
    selected=read_json(ROOT/'runs/formal/selected_checkpoint.json')
    audit=read_json(ROOT/'runs/data_audit.json')
    reload=read_json(ROOT/'runs/formal/reload_verified.json')
    preservation=assert_preserved()
    assert reload['reload_matches']
    assert before['provenance']==after['provenance']
    assert before['adapter_sha256']==sha256(PROJECT_ROOT/'runs/formal/adapter/adapter_model.safetensors')
    assert after['adapter_sha256']==sha256(ROOT/'runs/formal/adapter/adapter_model.safetensors')
    original=read_json(PROJECT_ROOT/'runs/formal/summary.json')
    assert original['settings']==train['settings']==load_config()['training']
    output=ROOT/'REPORT.md'
    if output.exists(): raise FileExistsError('Preserve the existing v2 report')
    text=[
      '# 第一轮改进报告：缺失信息、兜底与工具调用边界',
      '',
      '本轮结果来自真实本地GPU训练与原始逐条预测。未覆盖第一版数据、适配器、预测、报告或源码。两版适配器在同一套新测试集上重新评测；第一版旧15条测试不作为本轮独立验收。',
      '',
      f'本轮是部分改善：结构通过{before["schema_valid"]["correct"]}/26→{after["schema_valid"]["correct"]}/26，缺失空值与路径共同正确{before["boundaries"]["missing_null_and_paths_exact"]["correct"]}/8→{after["boundaries"]["missing_null_and_paths_exact"]["correct"]}/8，工具完整匹配{before["tool_exact"]["correct"]}/4→{after["tool_exact"]["correct"]}/4，兜底{before["fallback_exact"]["correct"]}/10→{after["fallback_exact"]["correct"]}/10。与此同时，字段正确数{before["field_accuracy"]["correct"]}/88→{after["field_accuracy"]["correct"]}/88，非工具请求上的调用尝试{before["tool_attempts_on_non_tool_requests"]}→{after["tool_attempts_on_non_tool_requests"]}。因此保留两版，不能称为全面提升或稳定生产方案。',
      '',
      '## 范围与公平对照',
      '',
      '- 只补强缺失金额/税率/税号的null及missing_fields、计算参数不足或冲突的兜底、未注册工具和无关请求。未知金额不是0；显式0仍有效。没有引入新模型、付费资源、输出修复或自动重试。',
      '- 训练集保留旧40条并新增48条，验证集保留旧8条并新增16条；新测试26条使用不同写法及农业/仓储等业务场景。所有数据虚构，由助手编写并用程序校验，用户人工复核仍待完成。',
      '- 两版使用相同基座、4-bit NF4双重量化、SYSTEM_PROMPT、BF16/FP16选择和贪心解码。V2直接复用根目录已批准的configs/experiment.json，无新增训练配置。',
      '- V2从基座重新训练，并非在第一版适配器上继续训练。数据量及优化步数增加，验证集也扩大；本实验不能单独归因于某一条样本或某一项改动。',
      '- 原始预测不去代码围栏、不补字段。程序拦截不能使错误预测变正确；两版都使用同一V2检查代码记录拦截，模型指标只按raw_output计算。',
      f'- 第一版保护校验通过：{preservation["v1_files_verified"]}个文件。新增数据集哈希、原适配器哈希、训练配置和基座清单哈希保存在data/manifest.json及各metrics.json。',
      f'- 规范化重复数0；跨集合字符相似度最大{audit["maximum_cross_split_similarity"]:.3f}，最近样本{audit["closest_pair"]}，预检阈值0.85。该自动检查只辅助排查，不证明语义上完全独立。',
      '',
      '## 新测试集同题比较',
      '',
      '| 指标 | 第一版适配器 | V2适配器 |',
      '|---|---|---|',
    ]
    for label,key in [('JSON解析','json_parse'),('完整结构通过','schema_valid'),('路由正确','routing'),('抽取字段准确','field_accuracy'),('工具名与完整参数正确','tool_exact'),('兜底动作与原因正确','fallback_exact')]:
        text.append(f'| {label} | {fraction(before[key])} | {fraction(after[key])} |')
    text.append(f'| 缺失空值及路径共同正确 | {fraction(before["boundaries"]["missing_null_and_paths_exact"])} | {fraction(after["boundaries"]["missing_null_and_paths_exact"])} |')
    text.append(f'| 缺失路径micro-F1 | {f1(before["missing_fields_f1"])} | {f1(after["missing_fields_f1"])} |')
    for label,key in [('空值处虚构字段次数','invented_values_on_null_fields'),('非工具请求上的调用尝试','tool_attempts_on_non_tool_requests'),('程序实际执行次数','runtime_executed'),('程序拦截次数','runtime_blocked')]:
        text.append(f'| {label} | {before[key]} | {after[key]} |')
    text.append(f'| 推理时延中位数/秒 | {before["latency_median_seconds"]:.2f} | {after["latency_median_seconds"]:.2f} |')
    text.extend(['','缺失空值及路径共同正确：仅统计8条missing_extract；要求结构通过、missing_fields集合正确，并保持所有期望未知值为null。它不要求其他非空字段也正确。工具完整匹配的分母是4条合法工具请求；兜底分母是10条请求。无效JSON仍留在各自分母。', '', '### 各场景完整响应正确数', '', '| 场景 | 第一版 | V2 |','|---|---|---|'])
    for name,left in before['boundaries']['scenario_exact'].items():
        right=after['boundaries']['scenario_exact'][name]
        text.append(f'| {name} | {left["correct"]}/{left["total"]} | {right["correct"]}/{right["total"]} |')
    text.extend(['','## 训练与重载证据','',f'正式训练{train["epochs_started"]}轮、{train["optimizer_steps"]}次参数更新，用时{train["training_seconds"]:.1f}秒，计算精度{train["dtype"]}。第一版为{original["optimizer_steps"]}次更新。',f'峰值allocated/reserved显存：{train["peak_allocated_mib"]:.0f}/{train["peak_reserved_mib"]:.0f} MiB。最大参数更新{train["maximum_parameter_update"]:.6f}，损失全部有限={train["all_losses_finite"]}。',f'按24条验证集平均答案损失最低选择第{selected["epoch"]}轮，验证损失{selected["dev_loss"]:.5f}。新进程离线重载贪心输出与保存的参考输出一致={reload["reload_matches"]}。','','| 轮次 | 验证损失 |','|---|---|'])
    for row in train['dev_losses']:text.append(f'| {row["epoch"]} | {row["loss"]:.5f} |')
    left_records=[json.loads(line) for line in (ROOT/'runs/test-before/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    right_records=[json.loads(line) for line in (ROOT/'runs/test-after/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    outcomes={'improved':[],'regressed':[],'still_failed':[]}
    for left,right in zip(left_records,right_records):
        assert left['id']==right['id'] and left['source_text']==right['source_text']
        lp,le=parsed(left);rp,re=parsed(right)
        lok=lp is not None and equal_nested(lp,left['expected']);rok=rp is not None and equal_nested(rp,right['expected'])
        key='improved' if rok and not lok else 'regressed' if lok and not rok else 'still_failed' if not rok else None
        if key:outcomes[key].append((left,right,le,re))
    text.extend(['','## 真实进步、退化与失败','',f'按完整响应匹配计：进步{len(outcomes["improved"])}条、退化{len(outcomes["regressed"])}条、两版都未完全正确{len(outcomes["still_failed"])}条。全部原始输出保存在runs/test-before和runs/test-after。'])
    for key,title in [('improved','改善案例'),('regressed','退化案例'),('still_failed','仍失败案例')]:
        selected_cases=outcomes[key][:2]
        if key=='still_failed':
            selected_cases += [case for case in outcomes[key] if case[1]['id'] in ('v2-test-023','v2-test-025') and case not in selected_cases]
        if not selected_cases:text.extend(['',f'### {title}','', '本轮没有出现此类完整响应变化，未编造案例。']);continue
        for left,right,le,re in selected_cases:
            text.extend(['',f'### {title} {right["id"]}','','```json',json.dumps({'input':right['source_text'],'expected':right['expected'],'before_raw':left['raw_output'],'after_raw':right['raw_output'],'before_guard':left['runtime_guard'],'after_guard':right['runtime_guard'],'before_schema_error':le,'after_schema_error':re},ensure_ascii=False,indent=2),'```'])
    text.extend(['','## 运行时规则改动及限制','','- 先校验JSON及强类型Schema；null与missing_fields不一致会失败。缺参数不能执行计算。','- 在原文出现未注册calculate_*名称时，不允许用已注册工具代替；原文明确指定另一个已注册工具时也会拦截。','- 原文必须能找到唯一、明确标注且与参数一致的数值；重复冲突值及同一参数里的“或/范围”候选会被拦截。','- 检查是固定文字模式，可能对复杂但合法的表达过度拦截，不是通用语义理解，也不验证真实税务资格。模型应自行输出fallback，拦截只作为额外防线。','','## 复现与结论边界','','25项契约测试在训练前通过。Windows使用PYTHONUTF8=0、PYTHONIOENCODING=utf-8；离线标志HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。模型和依赖仍在根目录，本版不重复下载。','已有实验数据、训练日志和预测文件受到覆盖保护；新训练需另建版本，不能在此冻结测试上继续调参。26条合成测试只能说明这些案例的情况，不能证明真实企业场景可靠。改善或退化均按上表如实解释，不以程序拦截替代模型能力。'])
    output.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('V2_REPORT_WRITTEN',output,flush=True)

if __name__=='__main__':main()
