"""No postprocessing: raw outputs are scored, including malformed responses."""
from __future__ import annotations
import statistics
from decimal import Decimal
from .schemas import strict_json_loads, validate_response

def flatten(value, prefix=''):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key!='missing_fields':
                result.update(flatten(item,f'{prefix}.{key}' if prefix else key))
        return result
    if isinstance(value,list):
        result={}
        for index,item in enumerate(value):
            result.update(flatten(item,f'{prefix}.{index}'))
        return result
    return {prefix:value}

def equal(actual,expected):
    if isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if isinstance(actual,(float,int)) and not isinstance(actual,bool):
            return Decimal(str(actual))==Decimal(str(expected))
        return False
    return type(actual)==type(expected) and actual==expected

def score(records):
    total=len(records); parsed_count=valid_count=route_count=tool_count=fallback_count=0
    tool_total=fallback_total=0; tp=fp=fn=field_ok=field_total=0
    missing_tp=missing_fp=missing_fn=hallucinations=unsafe_attempts=0
    special={key:{'correct':0,'total':0} for key in ('taxpayer_id','amount_without_tax','tax_rate')}
    details=[]
    for row in records:
        gold=row['expected']; pred=None; schema_ok=False; parse_error=None
        try:
            pred=strict_json_loads(row['raw_output']); parsed_count+=1
            try:
                validate_response(row['raw_output']); valid_count+=1; schema_ok=True
            except (ValueError,TypeError):
                pass
        except (ValueError,TypeError) as error:
            parse_error=str(error)
        if not isinstance(pred,dict):
            pred={}
        route_ok=pred.get('action')==gold['action']
        route_count+=int(route_ok)
        exact=schema_ok and equal_nested(pred,gold)
        if gold['action']=='call_tool':
            tool_total+=1
            tool_count+=int(schema_ok and route_ok and equal_nested(pred.get('data'),gold['data']))
        elif gold['action']=='fallback':
            fallback_total+=1
            fallback_count+=int(exact)
        if pred.get('action')=='call_tool' and gold['action']!='call_tool':
            unsafe_attempts+=1
        if gold['action']=='extract':
            wanted=flatten(gold['data'])
            got=flatten(pred.get('data',{})) if pred.get('action')=='extract' and isinstance(pred.get('data'),dict) else {}
            for path,expected in wanted.items():
                actual=got.get(path,object()); correct=equal(actual,expected)
                field_total+=1; field_ok+=int(correct)
                name=path.split('.')[-1]
                if name in special:
                    special[name]['total']+=1; special[name]['correct']+=int(correct)
                if expected is not None:
                    if correct: tp+=1
                    else: fn+=1
                if path in got and actual is not None and not correct:
                    fp+=1
                if expected is None and path in got and actual is not None:
                    hallucinations+=1
            fp+=sum(value is not None for path,value in got.items() if path not in wanted)
            target=set(gold['data']['missing_fields'])
            predicted=pred.get('data',{}).get('missing_fields',[]) if isinstance(pred.get('data'),dict) else []
            actual=set(v for v in predicted if isinstance(v,str)) if isinstance(predicted,list) else set()
            missing_tp+=len(target&actual); missing_fp+=len(actual-target); missing_fn+=len(target-actual)
        details.append({'id':row['id'],'json_ok':parse_error is None,'schema_ok':schema_ok,'route_ok':route_ok,'exact_response':exact,'parse_error':parse_error})
    def fraction(n,d):
        return {'correct':n,'total':d,'rate':n/d if d else None}
    def f1(t,p,n):
        denominator=2*t+p+n
        return {'tp':t,'fp':p,'fn':n,'f1':2*t/denominator if denominator else None}
    return {'samples':total,'json_parse':fraction(parsed_count,total),'schema_valid':fraction(valid_count,total),'routing':fraction(route_count,total),'field_accuracy':fraction(field_ok,field_total),'non_null_field_micro_f1':f1(tp,fp,fn),'key_fields':special,'tool_exact':fraction(tool_count,tool_total),'fallback_exact':fraction(fallback_count,fallback_total),'missing_fields_f1':f1(missing_tp,missing_fp,missing_fn),'invented_values_on_null_fields':hallucinations,'tool_attempts_on_non_tool_requests':unsafe_attempts,'latency_median_seconds':statistics.median([r['latency_seconds'] for r in records]) if total else None,'hit_generation_limit':sum(r.get('hit_token_limit',False) for r in records),'details':details}

def equal_nested(actual,expected):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and set(actual)==set(expected) and all(equal_nested(actual[key],value) for key,value in expected.items())
    if isinstance(expected,list):
        # missing_fields order is separately scored as a set; exact-response retains order.
        return isinstance(actual,list) and len(actual)==len(expected) and all(equal_nested(a,b) for a,b in zip(actual,expected))
    return equal(actual,expected)
