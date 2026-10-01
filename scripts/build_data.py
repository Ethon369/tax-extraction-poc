"""Auditable, fictional examples. Distinct writing formats are assigned to each split."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, SYSTEM_PROMPT, dumps, sha256, write_json
from src.schemas import validate_response

def tid(index):
    return f"TEST{index:014d}"

def item(name, amount, rate, tax=None):
    return dict(name=name, amount_without_tax=amount, tax_rate=rate, tax_amount=tax)

def extract(day, buyer, taxpayer, items):
    data = dict(invoice_date=day, taxpayer_id=taxpayer, buyer_name=buyer, items=items)
    missing = [key for key in ("invoice_date", "taxpayer_id", "buyer_name") if data[key] is None]
    if not items:
        missing.append("items")
    for i, row in enumerate(items):
        missing.extend(f"items.{i}.{key}" for key in ("amount_without_tax", "tax_rate") if row[key] is None)
    data["missing_fields"] = sorted(missing)
    return dict(action="extract", data=data, reason=None)

def call(name, **kwargs):
    return dict(action="call_tool", data=dict(tool_name=name, parameters=kwargs), reason=None)

def fallback(reason="unsupported_request"):
    return dict(action="fallback", data=None, reason=reason)

def build():
    sets = {name: [] for name in ("train", "dev", "test", "smoke")}
    def add(split, category, text, answer, family):
        validate_response(dumps(answer))
        index = len(sets[split]) + 1
        sets[split].append({"id": f"{split}-{index:02d}", "category": category, "template_family": family, "synthetic": True, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}, {"role": "assistant", "content": dumps(answer)}]})
    def ex(split, category, text, day, buyer, n, items, family):
        add(split, category, text, extract(day, buyer, None if n is None else tid(n), items), family)

    templates = [
        '请提取：开票日期{day}；购买方{buyer}；税号{id}；{name}，不含税金额{amount}元，税率{percent}%。',
        '增值税发票\n买方：{buyer}\n购买方税号：{id}\n日期：{day}\n项目：{name}，未税金额{amount}，税率{percent}%。',
        '单据信息：{day}向{buyer}开具{ name }发票，买方识别号{id}，净额{amount}元，税率{percent}%。',
        '录入这张票：货物{name}/未税{amount}元/税率{percent}%/开票{day}/购方{buyer}/购方税号{id}。',
        '发票抬头={buyer}，纳税人识别号={id}，开票日={day}；商品{name}；不含税价款{amount}；税率{percent}%。',
        '{buyer}的报销附件：票面日期{day}，税号{id}，明细{name}，金额为不含税{amount}元，税率{percent}%。仅提取。',
        '购买方[{buyer}] 税号[{id}] 日期[{day}]。请整理项目[{name}]，其未税价款为{amount}元、税率{percent}%。',
        '票据转录：日期 {day}；购货单位 {buyer}；购货税号 {id}；服务 {name}；未税金额 {amount}；税率 {percent}%。',
        '从下面提取JSON：买方{buyer}（{id}），{day}，{name}未税{amount}元，税率{percent}%。',
        '单项发票记录——抬头{buyer}，识别号{id}，开票日期{day}。品名{name}，不含税金额{amount}元，税率{percent}%。',
        '这张{day}的票请保存字段：买家{buyer}，税号{id}。收费项目{name}，未税金额{amount}元，税率{percent}%。',
        '财务录入卡：{buyer}｜{id}｜{day}；{name}：不含税金额{amount}元，税率{percent}%。',
        '嗯，帮我录一下，{buyer}那张票，税号{id}，开票{day}。买的{name}，没税的价是{amount}元，税率{percent}%，谢谢。',
        '别管旁边的电话12345，票上买方是{buyer}、税号{id}、日期{day}；{name}不含税金额{amount}，税率{percent}%。',
        '上次说错了，这次只看这里：开票{day}，{buyer}，税号{id}，{name}未税{amount}元、税率{percent}%。',
        '有点乱的OCR：购 方 {buyer}；税号{id}；开票{day}；明细{name}；不含税{amount}元；税率{percent}%。整理一下。',
        '报销聊天转述：同事说{day}给{buyer}开的票，买方号{id}，买{name}，未税价{amount}元，税率{percent}%。不用算。',
        '请忽略快递单号8888。真正发票买方{buyer}，税号{id}，时间{day}，货物{name}，不含税金额{amount}元，税率{percent}%。',
        '啊对了，票据内容是{buyer}/{id}，开票{day}，项目{name}不含税{amount}元，税率{percent}%。后面的“已报销”只是备注。',
        '财务你好，{buyer}，税号{id}，{day}；{name}的未税金额{amount}元、税率{percent}%。帮我结构化，不要写寒暄。',
    ]
    for i, template in enumerate(templates, 1):
        day = f"2026-08-{i:02d}"; buyer = f"模拟{i}号公司"; name = ["纸张", "电缆", "培训", "设备租赁"][i % 4]
        amount = 100 + i * 17; rate = [0.13, 0.06, 0.09, 0.01][i % 4]
        text = template.replace('{ name }', '{name}').format(day=day,buyer=buyer,id=tid(i),name=name,amount=amount,percent=round(rate*100))
        rows = [item(name, amount, rate)]
        if i in (4, 10, 18):
            text += '第二项是墨盒，不含税金额50元，税率13%，税额6.50元。'
            rows.append(item('墨盒',50,0.13,6.5))
        ex('train','standard' if i<=12 else 'noisy',text,day,buyer,i,rows,f'train-layout-{i:02d}')
    tax_prompts = [
        '帮我计算税额：不含税金额1000元，税率13%。',
        '请调用税额计算工具，未税金额240元，税率6%。',
        '这笔不含税价款750元，税率9%，求税额。',
        '我只要税金的计算指令：净额300元；征收率1%。',
        '执行calculate_tax_amount，不含税金额88.5元、税率13%。',
    ]
    for i,(text,a,r) in enumerate(zip(tax_prompts,[1000,240,750,300,88.5],[.13,.06,.09,.01,.13])):
        add('train','tool',text,call('calculate_tax_amount',amount_without_tax=a,tax_rate=r),f'train-tax-{i}')
    ded_prompts = [
        '已知税额130元，可抵扣比例100%，帮我算进项抵扣额。',
        '请生成抵扣计算调用：税额60元，抵扣比例50%。',
        '税金90元，允许抵扣比例80%，计算可抵扣数额。',
        '本次按给定条件算：税额27.5元，可抵扣份额20%。',
        '调用calculate_vat_deduction，税额99元，抵扣比例0%。',
    ]
    for i,(text,a,r) in enumerate(zip(ded_prompts,[130,60,90,27.5,99],[1,.5,.8,.2,0])):
        add('train','tool',text,call('calculate_vat_deduction',tax_amount=a,deductible_ratio=r),f'train-ded-{i}')
    ex('train','missing','提取发票：2026-08-25，模拟缺号公司，税号没写；纸张未税200元、税率13%。','2026-08-25','模拟缺号公司',None,[item('纸张',200,.13)],'train-missing-id')
    ex('train','missing',f'购买方模拟模糊公司，税号{tid(32)}，开票2026-08-26，办公桌未税金额看不清，税率13%。请提取。','2026-08-26','模拟模糊公司',32,[item('办公桌',None,.13)],'train-missing-amount')
    ex('train','missing',f'日期无法辨认。买方模拟无日公司，税号{tid(33)}。维修未税500元，税率6%，请整理。',None,'模拟无日公司',33,[item('维修',500,.06)],'train-missing-date')
    ex('train','missing',f'开票2026-08-28，模拟无率公司，税号{tid(34)}；电缆未税600元，税率未注明，只提取原文。','2026-08-28','模拟无率公司',34,[item('电缆',600,None)],'train-missing-rate')
    add('train','missing','帮我算抵扣额：税额100元，但是没提供抵扣比例。',fallback('insufficient_information'),'train-missing-ded-ratio')
    add('train','missing','想算税额，只知道税率13%，未税金额暂时没有。',fallback('insufficient_information'),'train-missing-tax-base')
    for i,text in enumerate(['北京明天会下雨吗？','请帮我算个人所得税。','讲个关于会计的笑话。','调用calculate_corporate_income_tax算企业所得税。']):
        add('train','fallback',text,fallback(),f'train-unsupported-{i}')

    ex('dev','standard',f'核验记录：收票单位模拟甲研究室；开具日2026-09-01；受票方编码{tid(101)}。采购清单仅电池，未税金额420元，税率13%，税额54.6元。','2026-09-01','模拟甲研究室',101,[item('电池',420,.13,54.6)],'dev-audit-card')
    ex('dev','noisy',f'附言写着“待审批”。从票面抄出：受票人模拟乙商店，识别号{tid(102)}，2026年9月2日开票；运输的税率为9%，未税金额170元。','2026-09-02','模拟乙商店',102,[item('运输',170,.09)],'dev-reversed-rate')
    ex('dev','standard',f'台账转文字：发票开于2026/09/03，抬头模拟丙厂，买方税号{tid(103)}。两行：螺钉未税80元税率13%；咨询未税120元税率6%。','2026-09-03','模拟丙厂',103,[item('螺钉',80,.13),item('咨询',120,.06)],'dev-ledger-two-lines')
    ex('dev','missing','票角破损导致税号遗失；其余读到购买单位模拟丁社，日期2026-09-04，保洁未税金额230元，税率6%。','2026-09-04','模拟丁社',None,[item('保洁',230,.06)],'dev-damaged-corner')
    add('dev','tool','建立计算请求，参数为：税率9%，不含税金额350元。目标是税额。',call('calculate_tax_amount',amount_without_tax=350,tax_rate=.09),'dev-parameter-order')
    add('dev','tool','按双方明确约定的抵扣比例25%，对税额40元执行抵扣金额计算。',call('calculate_vat_deduction',tax_amount=40,deductible_ratio=.25),'dev-ratio-first')
    add('dev','fallback','推荐一个适合旅行的城市。',fallback(),'dev-travel')
    add('dev','missing','只拿到了不含税金额680元，税率待核对，请给出税额计算指令。',fallback('insufficient_information'),'dev-pending-rate')

    test_specs = [
        (f'档案摘录（只做字段整理）：销售方模拟销售企业，购货方模拟星河公司，购方识别号{tid(201)}；签发日期2026-09-10。订购路由器，未税价值890元，适用税率13%，票面税额115.7元。','2026-09-10','模拟星河公司',201,[item('路由器',890,.13,115.7)],'test-seller-buyer'),
        (f'将收据逐栏转换：购买单位→模拟山川合作社；购买单位税号→{tid(202)}；开票日期→2026年09月11日；服务→设计；未税金额→560；税率→6%。','2026-09-11','模拟山川合作社',202,[item('设计',560,.06)],'test-arrow-form'),
        (f'有两笔明细需要保留顺序。模拟溪流实验室（买方号{tid(203)}）在2026-09-12收到发票：先为试剂，未税360元税率13%；后为检测，未税240元税率6%。','2026-09-12','模拟溪流实验室',203,[item('试剂',360,.13),item('检测',240,.06)],'test-order-preface'),
        (f'报销申请对应票面：购方模拟晴空工作室；号{tid(204)}；开票2026/9/13；软件订阅，不含税金额199.99元，税率6%，税额12元。提取票面数值即可。','2026-09-13','模拟晴空工作室',204,[item('软件订阅',199.99,.06,12)],'test-decimal-receipt'),
        (f'口述修正：不是订单日期，请用发票开具时间2026-09-14。购买方模拟松林公司，税号{tid(205)}。明细为住宿，未税金额480元、税率6%。','2026-09-14','模拟松林公司',205,[item('住宿',480,.06)],'test-date-correction'),
        (f'页眉流水号7009无需录入。正文写购买方模拟云帆中心（税号{tid(206)}），开票日期2026-09-15；印刷费未税金额320元，税率13%。页尾为经办人小陈。','2026-09-15','模拟云帆中心',206,[item('印刷费',320,.13)],'test-header-footer'),
        (f'按摘要还原：{tid(207)}是模拟竹叶商行的买方税号，票于2026-09-16开出。清单上的物流：税率9%，不含税金额660元。不要额外计算税额。','2026-09-16','模拟竹叶商行',207,[item('物流',660,.09)],'test-id-first'),
        ('请登记残缺票：模拟远山公司收到2026-09-17开具的纸箱发票，税号这一栏空白。未税金额210元，税率13%。','2026-09-17','模拟远山公司',None,[item('纸箱',210,.13)],'test-blank-id'),
        (f'扫描备注称金额“大约三四百”，没有确定数字；其余字段为2026-09-18、购买方模拟春雨社、税号{tid(209)}、品名茶叶、税率13%。请忠实抽取。','2026-09-18','模拟春雨社',209,[item('茶叶',None,.13)],'test-vague-range'),
        (f'请读这段未完整识别的票据：购买单位模拟晨光厂，识别号{tid(210)}；开票日无法确认，维修服务未税450元，税率空缺。',None,'模拟晨光厂',210,[item('维修服务',450,None)],'test-two-missing'),
    ]
    for i,(text,day,buyer,n,rows,family) in enumerate(test_specs):
        ex('test','missing' if i>=7 else ('noisy' if i in (4,5,6) else 'standard'),text,day,buyer,n,rows,family)
    add('test','tool','将此需求转为计算函数：税率13%；不含税价款125.5元；我希望得到税额。',call('calculate_tax_amount',amount_without_tax=125.5,tax_rate=.13),'test-function-request')
    add('test','tool','已确认可抵扣比例40%，凭证所列税额75元。请按上述给定比例安排抵扣计算。',call('calculate_vat_deduction',tax_amount=75,deductible_ratio=.4),'test-confirmed-proportion')
    add('test','tool','任务名称：计算增值税税额。输入字段：未税金额=999元，税率=0.06。',call('calculate_tax_amount',amount_without_tax=999,tax_rate=.06),'test-decimal-rate')
    add('test','fallback','换个话题，周末的天气预报是什么？',fallback(),'test-weekend-weather')
    add('test','fallback','请执行calculate_land_appreciation_tax，为我计算土地增值税。',fallback(),'test-land-tool')

    add('smoke','tool','试运行税额工具：不含税金额10元，税率1%。',call('calculate_tax_amount',amount_without_tax=10,tax_rate=.01),'smoke-tax-a')
    add('smoke','tool','环境检查，请按税额2元、抵扣比例50%生成抵扣调用。',call('calculate_vat_deduction',tax_amount=2,deductible_ratio=.5),'smoke-ded-a')
    add('smoke','fallback','写一首小诗。',fallback(),'smoke-poem')
    add('smoke','missing','计算税额，但是未税金额未知、税率也未知。',fallback('insufficient_information'),'smoke-no-parameters')
    add('smoke','tool','小样本测试，税率6%，不含税金额20元，计算税额。',call('calculate_tax_amount',amount_without_tax=20,tax_rate=.06),'smoke-tax-b')
    add('smoke','tool','测试抵扣：税额3元，允许抵扣比例100%。',call('calculate_vat_deduction',tax_amount=3,deductible_ratio=1),'smoke-ded-b')
    for split, rows in sets.items():
        write_json(ROOT/'data'/f'{split}.json',rows)
    write_json(ROOT/'data/manifest.json',{'description':'Fictional, assistant-authored examples; programmatically validated. User review pending.', 'counts':{key:len(value) for key,value in sets.items()},'sha256':{key:sha256(ROOT/'data'/f'{key}.json') for key in sets},'test_policy':'Frozen before model experiments; no final-test-driven tuning.'})
    print({key:len(value) for key,value in sets.items()})

if __name__ == '__main__':
    build()
