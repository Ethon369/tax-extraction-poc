"""Version 2: explicit, synthetic boundary examples; refuse to overwrite a freeze."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, PROJECT_ROOT, SYSTEM_PROMPT, dumps, read_json, sha256, write_json
from src.schemas import validate_response

def identifier(number):
    return f'TEST{number:014d}'

def item(name, amount, rate, tax=None):
    return {'name':name,'amount_without_tax':amount,'tax_rate':rate,'tax_amount':tax}

def extraction(day, buyer, tax_id, items):
    data={'invoice_date':day,'taxpayer_id':tax_id,'buyer_name':buyer,'items':items}
    missing=[key for key in ('invoice_date','taxpayer_id','buyer_name') if data[key] is None]
    if not items: missing.append('items')
    for index,line in enumerate(items):
        missing.extend(f'items.{index}.{key}' for key in ('amount_without_tax','tax_rate') if line[key] is None)
    data['missing_fields']=sorted(missing)
    return {'action':'extract','data':data,'reason':None}

def tool(name, **parameters):
    return {'action':'call_tool','data':{'tool_name':name,'parameters':parameters},'reason':None}

def fallback(reason='unsupported_request'):
    return {'action':'fallback','data':None,'reason':reason}

def build():
    if (ROOT/'data/manifest.json').exists():
        raise FileExistsError('Version 2 is already frozen; do not overwrite it')
    splits={name:copy.deepcopy(read_json(PROJECT_ROOT/'data'/f'{name}.json')) for name in ('train','dev','smoke')}
    for name,rows in splits.items():
        for row in rows:
            row['id']='v2-base-'+row['id']
            row['scenario']='v1_rehearsal'
    splits['test']=[]
    def add(split, category, scenario, text, answer, family):
        validate_response(dumps(answer))
        index=len(splits[split])+1
        splits[split].append({'id':f'v2-{split}-{index:03d}','category':category,'scenario':scenario,'template_family':f'v2-{split}-{family}','synthetic':True,'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':text},{'role':'assistant','content':dumps(answer)}]})
    def ex(split, scenario, text, day, buyer, number, lines, family):
        add(split,'missing_extract' if scenario!='complete_control' else 'complete_control',scenario,text,extraction(day,buyer,None if number is None else identifier(number),lines),family)

    # Sixteen distinct training layouts, not a repeated amount/name template.
    train_extract=[
      ('missing_id','缺页发票登记：2026-07-01，演练机房站采购机柜，未税720元，税率13%；购买方税号所在页丢失。只输出现有字段。','2026-07-01','演练机房站',None,[item('机柜',720,.13)],'missing-page'),
      ('missing_amount',f'金额被遮挡的记录：购方演练网关室，编码{identifier(401)}，开具于2026-07-02。项目交换机；税率13%；未税金额读不到，不可按税额反推。','2026-07-02','演练网关室',401,[item('交换机',None,.13)],'covered-base'),
      ('missing_rate',f'维修工单附票需要抽取。发票日期2026-07-03，买方演练配线班（税号{identifier(402)}）；布线的未税金额350元，税率仍为空。','2026-07-03','演练配线班',402,[item('布线',350,None)],'work-order'),
      ('multiple_missing','残片上仅见：2026-07-04、购买方演练计量组、项目电表；税号和不含税价款均未识别，税率13%。请保留未知字段。','2026-07-04','演练计量组',None,[item('电表',None,.13)],'fragment'),
      ('multiple_missing',f'演练遥测室，识别号{identifier(403)}，开票日期2026-07-05。遥测服务的税率未知，不含税金额也没有提供；目标是登记而非计算。','2026-07-05','演练遥测室',403,[item('遥测服务',None,None)],'register-not-calculate'),
      ('missing_rate',f'两行明细摘抄：演练供电所，{identifier(404)}，2026-07-06。先列插座：未税90元，税率13%；后列巡检：未税160元，税率未录。请保持顺序。','2026-07-06','演练供电所',404,[item('插座',90,.13),item('巡检',160,None)],'second-line-rate'),
      ('missing_amount',f'需要字段化的采购摘要：2026-07-07开票给演练终端队，税号{identifier(405)}；第一项电源模块，价格没有抄到，税率13%；第二项线槽，未税42元，税率13%。','2026-07-07','演练终端队',405,[item('电源模块',None,.13),item('线槽',42,.13)],'first-line-base'),
      ('missing_id','核对一张票而不补全：演练负荷组为买方，日期2026-07-08，负荷测量未税金额250元，税率6%。旁边的电话12345678不是买方税号，税号栏没有内容。','2026-07-08','演练负荷组',None,[item('负荷测量',250,.06)],'phone-not-id'),
      ('missing_amount',f'请把这条口述转成JSON：演练电池组在2026-07-09收到蓄电池发票，税号{identifier(406)}；税率13%，未税金额只记得大概几百元。','2026-07-09','演练电池组',406,[item('蓄电池',None,.13)],'vague-price'),
      ('multiple_missing','票据映射任务：抬头演练传感组，开具2026-07-10，探头税率13%；没有清晰的税号，没有确定未税价款。不能用0表示这些缺失。','2026-07-10','演练传感组',None,[item('探头',None,.13)],'explicit-not-zero'),
      ('missing_id','已确认字段只有日期2026-07-11、受票单位演练继保室、商品继电器、未税金额0元、税率0%。税号未留；请忠实保留两个零值。','2026-07-11','演练继保室',None,[item('继电器',0,0)],'zero-versus-null'),
      ('multiple_missing',f'仪表附件摘要：买方演练仪表班、买方号{identifier(407)}、项目电压表。发票日期不可辨，未税价格不可辨，税率13%。请抽取可见信息。',None,'演练仪表班',407,[item('电压表',None,.13)],'date-and-base'),
      ('missing_rate',f'采购票上的报价不是税率。演练箱变组，编号{identifier(408)}，2026-07-13，箱体未税890元；备注订单量6件，税率缺失。只登记。','2026-07-13','演练箱变组',408,[item('箱体',890,None)],'quantity-not-rate'),
      ('missing_amount',f'扫描结果请整理：演练检修队；税号{identifier(409)}；开票2026-07-14；检修；税率6%；票面税额18元，未税价款丢失。不要从税额推算未税金额。','2026-07-14','演练检修队',409,[item('检修',None,.06,18)],'no-reverse-calculation'),
      ('multiple_missing','演练台区组的2026-07-15票据录入：通信卡未税20元、税率空白；安装未税金额未知、税率6%。购买方税号未显示。','2026-07-15','演练台区组',None,[item('通信卡',20,None),item('安装',None,.06)],'mixed-missing-paths'),
      ('missing_rate',f'抽取这张部分完整的凭证：发票给演练定位室，{identifier(410)}，2026-07-16，定位服务未税300元，税额已抄为18元，税率本身没有抄下，不能反推。','2026-07-16','演练定位室',410,[item('定位服务',300,None,18)],'rate-not-derived'),
    ]
    for args in train_extract: ex('train',*args)
    valid_tools=[
      ('以明确参数建立税额调用，不含税金额0元、税率13%，零金额不是缺失。','calculate_tax_amount',{'amount_without_tax':0,'tax_rate':.13}),
      ('这次征收率0%，净额560元，请计算税金。','calculate_tax_amount',{'amount_without_tax':560,'tax_rate':0}),
      ('数值已审核：税率0.06；未税金额81.25元。生成税额计算请求。','calculate_tax_amount',{'amount_without_tax':81.25,'tax_rate':.06}),
      ('请运行calculate_tax_amount，计税金额=640元，税率=9%。','calculate_tax_amount',{'amount_without_tax':640,'tax_rate':.09}),
      ('可抵扣比例0%，税额33元，请按显式条件计算，不是缺参数。','calculate_vat_deduction',{'tax_amount':33,'deductible_ratio':0}),
      ('按给出的可抵扣份额100%和税金24元，输出抵扣计算函数。','calculate_vat_deduction',{'tax_amount':24,'deductible_ratio':1}),
      ('仅做数值计算：税额=12.5元，抵扣比例=0.4。','calculate_vat_deduction',{'tax_amount':12.5,'deductible_ratio':.4}),
      ('希望得到calculate_vat_deduction调用。允许抵扣比例25%，税额48元，条件都已明确。','calculate_vat_deduction',{'tax_amount':48,'deductible_ratio':.25}),
    ]
    for index,(text,name,parameters) in enumerate(valid_tools): add('train','tool','valid_tool',text,tool(name,**parameters),f'valid-{index}')
    insufficient=[
      '生成税额计算调用；不含税金额170元，但税率完全未提供。',
      '税率9%可以确认，可惜未税价款还没有数值。现在需要计算税金。',
      '税额18元，请给出抵扣计算指令；抵扣比例未知，不得假设为100%。',
      '抵扣比例50%已明确，税额仍待补充，帮我计算可抵扣额。',
      '只提供一张金额为空、税率也空白的单据，却要直接计算税额。',
      '计算进项抵扣：未税金额500元，税率6%；没有直接提供税额和抵扣比例，不要推算参数。',
      '计算税额的草稿中同时写了不含税金额100元与不含税金额200元，税率13%，没有说明哪个有效。',
      '现需税额计算：未税金额80元；税率6%或13%，尚未选择。',
      '这份抵扣请求含税额10元以及税额20元，可抵扣比例50%，两笔数值指向同一项目但未确认。',
      '税额50元；抵扣比例20%或80%。请先生成抵扣计算指令，比例未定。',
      '计算税额，不含税金额300或400元，税率9%，未税金额还在核对。',
      '抵扣任务：税額数字被涂掉，仅可确认可抵扣比例0.25。',
    ]
    for index,text in enumerate(insufficient): add('train','fallback','insufficient_information',text,fallback('insufficient_information'),f'insufficient-{index}')
    unsupported=[
      '请调用calculate_stamp_duty，金额600元，算印花税；不要换成其他工具。',
      '未税金额100元，税率13%，但这次只要求calculate_property_tax工具。',
      '工具需求清单仅一个名字：calculate_vehicle_tax。目前系统不提供它，也不能用税额工具替代。',
      '税额60元，抵扣比例50%，请交给calculate_customs_tax处理。',
      '生成calculate_social_insurance_payment的调用，参数以后补。',
      '把calculate_personal_income_tax作为工具名输出，工资9000元。',
      '与票据无关：介绍一下操作系统的进程和线程。',
      '我想要一段晨会开场白，包含数字100和13%。',
      '帮我修改一封道歉信的语气，不需要处理发票。',
      '把“hello world”翻译成中文。',
      '请规划今天的健身训练，持续45分钟。',
      '写一个关于星空的谜语，税率13%只是随手打的字符。',
    ]
    for index,text in enumerate(unsupported): add('train','fallback','unsupported_request',text,fallback(),f'unsupported-{index}')

    # Additional validation layouts and business settings differ from training.
    dev_extract=[
      ('missing_id','校验纸质附件：开票2026-06-01，样例展馆购展板，未税145元、税率13%。只有购买方税号一栏被裁掉。','2026-06-01','样例展馆',None,[item('展板',145,.13)],'cropped-column'),
      ('missing_amount',f'从审阅意见重建字段：样例舞台社，{identifier(501)}，日期2026-06-02，灯架，税率13%；意见明确说未税金额未录入。','2026-06-02','样例舞台社',501,[item('灯架',None,.13)],'review-comments'),
      ('missing_rate',f'分栏摘记送审：购货方样例画室（{identifier(502)}），开具日2026-06-03，颜料净额260元；税率的位置显示“待补”，无需计算。','2026-06-03','样例画室',502,[item('颜料',260,None)],'pending-cell'),
      ('multiple_missing','审计人员只读出2026-06-04、样例乐团和乐谱项目；未税金额与税号均无法确认，税率13%。请输出部分结构而非拒绝抽取。','2026-06-04','样例乐团',None,[item('乐谱',None,.13)],'auditor-partial'),
      ('missing_rate',f'样例剧场的票面概要，编号{identifier(503)}，2026-06-05：布景未税510元、税率未识别；灯光调试未税70元、税率6%。只处理现有文本。','2026-06-05','样例剧场',503,[item('布景',510,None),item('灯光调试',70,.06)],'first-of-two-rate'),
    ]
    for args in dev_extract: ex('dev',*args)
    for index,(text,name,parameters) in enumerate([
      ('对参数列表进行工具映射：未税金额=72元；税率=13%；目标=税額。','calculate_tax_amount',{'amount_without_tax':72,'tax_rate':.13}),
      ('条件卡已填：税额0元，可抵扣比例60%。安排抵扣计算。','calculate_vat_deduction',{'tax_amount':0,'deductible_ratio':.6}),
      ('提交一个可执行税金计算请求，净额90元、征收率0%。','calculate_tax_amount',{'amount_without_tax':90,'tax_rate':0}),
    ]): add('dev','tool','valid_tool',text,tool(name,**parameters),f'mapping-{index}')
    for index,text in enumerate([
      '要得到税额；表格只有未税金额220元，税率格标记未核实。',
      '已知抵扣比例75%，税额信息缺页，是否能输出抵扣调用？',
      '请处理计算申请：不含税金额310元，税率1%或者6%，选择还未确认。',
      '抵扣申请记录税额15元、抵扣比例10%，另处又标抵扣比例90%，没有最终版本。',
    ]): add('dev','fallback','insufficient_information',text,fallback('insufficient_information'),f'pending-{index}')
    for index,text in enumerate([
      '不含税金额70元，税率13%；所需接口名却是calculate_environment_tax，不能更换。',
      '接口calculate_resource_tax要怎么调用？这里要求的是资源税业务。',
      '请给一本小说写读后感，题材是航海。',
      '整理我的周末家务安排，完全不涉及票据。',
    ]): add('dev','fallback','unsupported_request',text,fallback(),f'out-of-scope-{index}')

    # Holdout: new layouts AND different business scenarios; not old test reuse.
    test_extract=[
      ('missing_id','仓储归档员的交接说明：样例冷库为受票方，票开于2026-05-11；隔热板一项未税价款410元、税率13%。身份编码没有随附件交接，只登记已知字段。','2026-05-11','样例冷库',None,[item('隔热板',410,.13)],'handover-missing-id'),
      ('missing_amount',f'供应商回信仅补了税率13%。原来的票据抬头样例温室园、购方编码{identifier(601)}、开具时间2026-05-12、品名育苗盘都还在；未税金额依旧留白。请合并成抽取结果。','2026-05-12','样例温室园',601,[item('育苗盘',None,.13)],'supplier-reply'),
      ('missing_rate',f'报销人拍照时漏了税率所在区域。可读正文：2026-05-13给样例养殖场（{identifier(602)}）开具饲料发票，未税费用680元。将缺失也一并登记。','2026-05-13','样例养殖场',602,[item('饲料',680,None)],'photo-cropped-region'),
      ('multiple_missing','这次不是计算申请，而是补录：买方样例果园站，2026-05-14购买滴灌管；识别号未拿到，未税价款未拿到，税率13%已明确。','2026-05-14','样例果园站',None,[item('滴灌管',None,.13)],'backfill-not-calculation'),
      ('multiple_missing',f'缺项清单指出喷灌服务没有金额也没有税率；其发票头部仍完整：样例苗圃、税号{identifier(603)}、2026-05-15。请转换文字并列出缺项路径。','2026-05-15','样例苗圃',603,[item('喷灌服务',None,None)],'missing-item-checklist'),
      ('missing_rate',f'附件目录按顺序列两项，样例渔场/{identifier(604)}/2026-05-16：一是渔网，未税210元，税率13%；二是池塘维护，未税340元，税率栏没拍到。保持原序输出。','2026-05-16','样例渔场',604,[item('渔网',210,.13),item('池塘维护',340,None)],'attachment-directory'),
      ('missing_id','便签上的号码87654321只是快递编号。实际票据属于样例蜂场，开票2026-05-17，蜂箱未税金额190元、税率13%；买方识别号没有填写。请别把便签号码录成税号。','2026-05-17','样例蜂场',None,[item('蜂箱',190,.13)],'sticky-note-decoy'),
      ('multiple_missing',f'标签脱落的单据还剩下购买方样例茶园、买方码{identifier(605)}和茶叶烘焙项目，税率6%；开票日期与未税金额均未能恢复。按残存文字进行抽取。',None,'样例茶园',605,[item('茶叶烘焙',None,.06)],'lost-label'),
      ('complete_control',f'完整记录用于核对零值：样例种子站的发票日期2026-05-19，税号{identifier(606)}；种子样品未税金额0元、税率0%，票面税额0元。这些都是明确数字。','2026-05-19','样例种子站',606,[item('种子样品',0,0,0)],'explicit-zero-control'),
      ('complete_control',f'农机租用的签收附件已齐全，要求字段化：买方样例农机社，开票日2026-05-20，纳税人码{identifier(607)}；农机租用未税128元，税率6%，税额7.68元。','2026-05-20','样例农机社',607,[item('农机租用',128,.06,7.68)],'signed-attachment'),
      ('complete_control',f'请对电子摘要逐项录入，不补算税额：发票购买方样例粮站，税号{identifier(608)}，2026-05-21；包装袋不含税金额315.5元，税率13%。','2026-05-21','样例粮站',608,[item('包装袋',315.5,.13)],'electronic-summary'),
      ('complete_control',f'运输回单附票的已知信息为：购方样例牧场（{identifier(609)}）、2026-05-22、牧草运输、未税金额820元、税率9%。只需抽取这份完整信息。','2026-05-22','样例牧场',609,[item('牧草运输',820,.09)],'transport-receipt'),
    ]
    for args in test_extract: ex('test',*args)
    for index,(text,name,parameters) in enumerate([
      ('清算系统等待税额函数参数：税率13%和不含税金额67.5元均已确定，请输出调用。','calculate_tax_amount',{'amount_without_tax':67.5,'tax_rate':.13}),
      ('在抵扣计算入口，录入税额52元、可抵扣比例75%。这两个数值是本次唯一条件。','calculate_vat_deduction',{'tax_amount':52,'deductible_ratio':.75}),
      ('向税额计算工具提交零税率场景：不含税金额230元，税率0%，请执行金额计算。','calculate_tax_amount',{'amount_without_tax':230,'tax_rate':0}),
      ('数值演示要求使用calculate_vat_deduction：税额16元，可抵扣份额0%，请保留显式零比例。','calculate_vat_deduction',{'tax_amount':16,'deductible_ratio':0}),
    ]): add('test','tool','valid_tool',text,tool(name,**parameters),f'holdout-executable-{index}')
    insufficient_test=[
      ('missing_calculation_rate','准备提交税额计算任务时，未税金额记录为460元；税率仍在问供应商，没有任何确定值。现在先生成响应。'),
      ('missing_calculation_amount','计算税金所需的税率6%已回传，但不含税金额那一项没有回传，禁止从其他信息估算。'),
      ('missing_deduction_ratio','页面希望显示可抵扣金额，现有条件只有税额28元；抵扣比例尚无约定，不能默认全额抵扣。'),
      ('missing_deduction_tax','抵扣比例30%已写入条件，税额栏却显示无法识别；请回应这项抵扣计算请求。'),
      ('conflicting_rate','税额计算审批单有未税金额380元，税率一栏写9%或13%，审批人尚未勾选任何一个。'),
      ('conflicting_amount','要计算税额，但两个版本都未确认：旧页不含税金额150元，新页不含税金额250元；税率13%。'),
    ]
    for index,(scenario,text) in enumerate(insufficient_test): add('test','fallback',scenario,text,fallback('insufficient_information'),f'holdout-pending-{index}')
    unsupported_test=[
      ('unregistered_tool','请勿替换我要的接口：calculate_deed_tax。不含税金额360元、税率3%，这些数字供契税工具使用。'),
      ('unregistered_tool','集成方只接受calculate_city_maintenance_tax这个工具名；这里给税额40元、可抵扣比例50%，不要调用抵扣工具。'),
      ('unrelated_request','发票任务暂停，改为解释为什么树叶会变黄；数字13%与100元只是我粘贴错的内容。'),
      ('unrelated_request','请帮我为同学的毕业典礼写祝福，长度大约100字；这次没有单据处理需求。'),
    ]
    for index,(scenario,text) in enumerate(unsupported_test): add('test','fallback',scenario,text,fallback(),f'holdout-unsupported-{index}')
    assert len(splits['train'])==88 and len(splits['dev'])==24 and len(splits['test'])==26
    from check_data import audit_rows
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(PROJECT_ROOT/'models/Qwen2.5-1.5B-Instruct',local_files_only=True)
    audit_rows(splits,tokenizer)  # validate holdout separation and token limits before freezing
    protected=[]
    for directory in ('data','runs','src','configs'):
        protected.extend(p for p in (PROJECT_ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'))
    protected.append(PROJECT_ROOT/'REPORT.md')
    write_json(ROOT/'v1_preservation.json',{'files':{str(p.relative_to(PROJECT_ROOT)).replace('\\','/'):sha256(p) for p in protected},'original_git_commit':'d373bd8cf07301676c0894bca47daf93b7a05ce6'})
    for split,rows in splits.items(): write_json(ROOT/'data'/f'{split}.json',rows)
    write_json(ROOT/'data/manifest.json',{'version':'v2','counts':{s:len(rows) for s,rows in splits.items()},'sha256':{s:sha256(ROOT/'data'/f'{s}.json') for s in splits},'test_policy':'New holdout frozen before training and prediction; original final test excluded. No tuning after final testing.','prompt_policy':'Exactly the original SYSTEM_PROMPT; same prompt for before and after.','human_review':'Assistant-authored synthetic labels validated by code; user review pending.','added_train_examples':48,'added_dev_examples':16})
    print('V2_DATA_FROZEN', {s:len(rows) for s,rows in splits.items()}, flush=True)

if __name__=='__main__': build()
