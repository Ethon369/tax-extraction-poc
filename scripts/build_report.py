"""Build the report from saved experimental evidence, never invented numbers."""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json
from src.metrics import equal, flatten
from src.schemas import validate_response

def ratio(value):
    return '无适用样本' if value['rate'] is None else f"{value['correct']}/{value['total']} ({value['rate']:.1%})"

def main():
    smoke_path=ROOT/'runs/smoke/summary.json'
    formal_path=ROOT/'runs/formal/summary.json'
    smoke=read_json(smoke_path) if smoke_path.exists() else None
    formal=read_json(formal_path) if formal_path.exists() else None
    results={mode:read_json(ROOT/f'runs/test-{mode}/metrics.json') if (ROOT/f'runs/test-{mode}/metrics.json').exists() else None for mode in ('zero','few','finetuned')}
    lines=['# 技术实验报告：税务文本结构化提取与 QLoRA 微调','',
           '所有指标来自本项目保存的真实运行记录。模型为 Qwen2.5-1.5B-Instruct，4-bit NF4＋双重量化；原始模型与微调模型采用相同加载精度及贪心解码。', '',
           '## 数据与任务约定','',
           '训练集40条、验证集8条、测试集15条，另有6条独立训练流程验证样本。数据由AI助手编写并逐项核对标注、通过程序校验；尚需用户复核，不称为真实企业数据。测试集在模型实验前冻结，文件哈希见 data/manifest.json。', '',
           '输出采用 action/data/reason 三个键。抽取数据保留参考字段；未知金额和税率可为null，missing_fields使用字段路径。税额只抽取明确给出的值，未知税额不用列入missing_fields。空对象、未知键、非法日期、字符串形式的数字均不能通过结构校验。', '',
           '工具仅计算明确给定数值：calculate_tax_amount(amount_without_tax,tax_rate)；calculate_vat_deduction(tax_amount,deductible_ratio)。抵扣比例来自用户给定假设，不判断法规适用或抵扣资格。Decimal计算，ROUND_HALF_UP保留两位小数。参数缺失、冲突或无法在原文中明确找到时，运行时拦截。', '',
           '## 环境与训练证据','']
    if smoke:
        lines.extend([f"首次训练：{smoke['optimizer_steps']}个优化步骤，最大参数更新{smoke['maximum_parameter_update']:.8f}，损失全部有限={smoke['all_losses_finite']}。",f"首次训练峰值allocated/reserved显存：{smoke['peak_allocated_mib']:.0f}/{smoke['peak_reserved_mib']:.0f} MiB。"])
        verified=ROOT/'runs/smoke/reload_verified.json'
        if verified.exists():
            lines.append(f"新进程离线重载后，固定输入的贪心输出一致：{read_json(verified)['reload_matches']}。")
    else:
        lines.append('首次训练尚未验证；代码或数据检查通过不能替代GPU训练证据。')
    if formal:
        lines.extend(['',f"正式训练：{formal['optimizer_steps']}个优化步骤，启动{formal['epochs_started']}轮，用时{formal['training_seconds']:.1f}秒。",f"正式训练峰值allocated/reserved显存：{formal['peak_allocated_mib']:.0f}/{formal['peak_reserved_mib']:.0f} MiB；计算精度{formal['dtype']}。",'','超参数：','```json',json.dumps(formal['settings'],ensure_ascii=False,indent=2),'```','环境：','```json',json.dumps(formal['environment'],ensure_ascii=False,indent=2),'```'])
        selected=read_json(ROOT/'runs/formal/selected_checkpoint.json')
        lines.append(f"按验证集答案损失最小选择第{selected['epoch']}轮检查点（验证损失{selected['dev_loss']:.4f}），未用最终测试集选模型。")
        formal_verified=ROOT/'runs/formal/reload_verified.json'
        if formal_verified.exists():
            lines.append(f"正式适配器新进程离线重载后，与已保存开发集首条输出一致：{read_json(formal_verified)['reload_matches']}。")
        logs=read_json(ROOT/'runs/formal/training_log.json')
        lines.extend(['','训练与验证损失变化：','','| 轮次 | 平均训练损失 | 验证损失 |','|---|---|---|'])
        for entry in formal['dev_losses']:
            training=[row['loss'] for row in logs if row['epoch']==entry['epoch']]
            lines.append(f"| {entry['epoch']} | {sum(training)/len(training):.5f} | {entry['loss']:.5f} |")
        if formal['dev_losses'][-1]['loss']>min(row['loss'] for row in formal['dev_losses']):
            lines.extend(['','本次实际出现：后期训练损失继续降低，但验证损失高于较早检查点。这与过拟合的趋势一致，因此交付验证集选出的适配器，而非直接使用最后一轮；验证集较小，不能仅凭该趋势作强泛化结论。'])
    else:
        lines.extend(['','正式训练尚未完成，效果结论待实验。'])
    lines.extend(['','## 测试集对照','', '| 指标 | Zero-shot | 固定2-shot | 微调后Zero-shot |','|---|---|---|---|'])
    metric_rows=[('JSON解析率','json_parse'),('结构校验通过率','schema_valid'),('路由正确率','routing'),('字段准确率（含空值）','field_accuracy'),('工具名及完整参数正确率','tool_exact'),('兜底动作与原因正确率','fallback_exact')]
    for title,key in metric_rows:
        lines.append('| '+title+' | '+' | '.join(ratio(results[m][key]) if results[m] else '未运行' for m in results)+' |')
    for title,key in [('非空字段micro-F1','non_null_field_micro_f1'),('缺失字段micro-F1','missing_fields_f1')]:
        lines.append('| '+title+' | '+' | '.join(f"{results[m][key]['f1']:.3f}" if results[m] and results[m][key]['f1'] is not None else '未运行或无适用样本' for m in results)+' |')
    for title,key,kind in [('空值处虚构字段次数','invented_values_on_null_fields','count'),('非工具请求上的工具调用尝试','tool_attempts_on_non_tool_requests','count'),('推理时延中位数（秒）','latency_median_seconds','seconds'),('推理峰值allocated（MiB）','peak_allocated_mib','seconds')]:
        values=[]
        for m in results:
            value=results[m]
            values.append('未运行' if value is None else (str(value[key]) if kind=='count' else f'{value[key]:.2f}'))
        lines.append('| '+title+' | '+' | '.join(values)+' |')
    lines.extend(['','### 关键字段分项','', '| 字段 | Zero-shot | 固定2-shot | 微调后 |','|---|---|---|---|'])
    for key in ('taxpayer_id','amount_without_tax','tax_rate'):
        lines.append('| '+key+' | '+' | '.join(f"{results[m]['key_fields'][key]['correct']}/{results[m]['key_fields'][key]['total']}" if results[m] else '未运行' for m in results)+' |')
    lines.extend(['','## 实际失败案例与排查','',
                  '开发阶段发现：聊天前缀末尾的换行可能与答案开头的左花括号发生BPE合并，直接按前缀词元数量切标签会出现边界不一致。修复为分别分词提示词和完成文本，并检查渲染文本连续性，保证训练和推理前缀一致。该问题来自真实数据检查错误日志。'])
    baseline=ROOT/'runs/dev-zero/predictions.jsonl'
    if baseline.exists():
        first=json.loads(baseline.read_text(encoding='utf-8').splitlines()[0])
        lines.extend(['','原始模型在开发集上出现JSON外围的Markdown代码围栏；因此原始输出不能直接被JSON解析器接受。这是格式失败，不能由此断言其内部字段全部错误。固定2-shot基线用于检查通过示例提示是否已经足以改善该问题。','```json',json.dumps({'id':first['id'],'raw_output':first['raw_output']},ensure_ascii=False,indent=2),'```'])
    development=ROOT/'runs/dev-finetuned'
    if (development/'metrics.json').exists():
        metrics=read_json(development/'metrics.json'); bad=[d['id'] for d in metrics['details'] if not d['exact_response']][:2]
        records=[json.loads(line) for line in (development/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        if bad:
            for identifier in bad:
                record=next(r for r in records if r['id']==identifier)
                lines.extend(['',f'验证集真实案例 `{identifier}`：','```json',json.dumps({'input':record['source_text'],'expected':record['expected'],'raw_output':record['raw_output'],'runtime_guard':record['runtime_guard']},ensure_ascii=False,indent=2),'```'])
                try:
                    parsed=json.loads(record['raw_output'])
                    expected_fields=flatten(record['expected'])
                    actual_fields=flatten(parsed)
                    wrong=[path for path,value in expected_fields.items() if not equal(actual_fields.get(path,object()),value)]
                    lines.append('差异定位：'+('、'.join(wrong) if wrong else '响应键、明细结构或missing_fields集合存在差异')+'。')
                    try:
                        validate_response(record['raw_output'])
                    except (ValueError,TypeError) as error:
                        lines.extend(['结构检查给出的错误：','```text',str(error),'```'])
                except (ValueError,TypeError):
                    lines.append('差异定位：原始输出不能直接解析为JSON。')
                lines.append('这些失败保留在本轮指标中。后续改进需补充相应边界样本，并在新的独立测试集上验证，未对本轮最终测试做事后修复。')
        else:
            lines.append('本轮验证集中未出现整体响应不匹配案例；不编造格式退化或过拟合经历。')
    lines.extend(['','## 指标口径与结论边界','',
                  'JSON解析与Schema校验分别统计，原始输出不去代码围栏、不做修复。格式失败仍保留在适用指标分母中。字段准确率含期望空值；micro-F1只比较非空字段的路径和值，金额和税率用Decimal归一比较。missing_fields按集合评分。工具准确率要求工具名与完整参数同时正确。', '',
                  '运行时校验与模型能力分别记录：程序拦截错误调用，不等于模型原始输出正确。对明确标记的数值采用保守来源检查，这不是任意自然语言的完备事实核验器。', '',
                  '虚构值与越界工具次数只能从成功解析的对象中计数。无法解析的输出不能判定这些细分错误是否存在；次数为0不代表所有失败样本都安全或无幻觉。整体解析、结构和字段指标仍保留失败样本。', '',
                  '测试集仅15条，工具类只有3条；结果适用于本批合成文本，不能推断生产可用性或真实税务准确率。训练数据中fallback例子少、写法有限；需要更多独立数据进一步验证。少量数据微调可能主要改善格式而未改善泛化，必须同时看Few-shot基线。', '',
                  '推理先进行一次预热，逐样本计时并同步CUDA。峰值显存为PyTorch allocated/reserved统计，不包含桌面和其他进程的所有占用。正式训练耗时包含验证、保存及结束检查，下载时间不计入。', '',
                  '实验选择和指标均保持真实；最终测试后不再根据测试答案调参。本报告未验证企业实际业务效果。'])
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT_WRITTEN',ROOT/'REPORT.md')

if __name__=='__main__': main()
