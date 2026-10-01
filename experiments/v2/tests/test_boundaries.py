import copy
import json
import unittest
from src.common import dumps, read_split
from src.schemas import validate_response
from src.tools import execute_checked
from src.metrics import score
from src.boundary_metrics import boundary_score

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.call={'action':'call_tool','data':{'tool_name':'calculate_tax_amount','parameters':{'amount_without_tax':100,'tax_rate':.13}},'reason':None}

    def test_unregistered_source_cannot_be_substituted(self):
        for source in ('请运行calculate_deed_tax，不含税金额100元，税率13%。','不含税金额100元，税率13%，但只用calculate_unregistered。'):
            self.assertEqual(execute_checked(dumps(self.call),source)['reason'],'unregistered_tool_in_request')

    def test_wrong_registered_tool_cannot_replace_requested_one(self):
        source='calculate_vat_deduction；税额10元，抵扣比例50%；不含税金额100元、税率13%。'
        self.assertEqual(execute_checked(dumps(self.call),source)['reason'],'requested_tool_mismatch')

    def test_unlabelled_alternative_rate_blocked(self):
        self.assertEqual(execute_checked(dumps(self.call),'不含税金额100元，税率13%或6%，未确认。')['status'],'blocked')

    def test_unlabelled_amount_range_blocked(self):
        for source in ('不含税金额100或200元，税率13%。','不含税金额100-200元，税率13%。'):
            self.assertEqual(execute_checked(dumps(self.call),source)['status'],'blocked')

    def test_each_missing_calculation_slot_blocked(self):
        for source in ('不含税金额100元，税率未知。','未税金额未提供，税率13%。','什么参数都没有，求税额。'):
            self.assertEqual(execute_checked(dumps(self.call),source)['status'],'blocked')

    def test_zero_is_explicit_and_executable(self):
        response=copy.deepcopy(self.call);response['data']['parameters']['amount_without_tax']=0
        self.assertEqual(execute_checked(dumps(response),'不含税金额0元，税率13%。')['result_yuan'],'0.00')
        response['data']['parameters']={'amount_without_tax':100,'tax_rate':0}
        self.assertEqual(execute_checked(dumps(response),'不含税金额100元，税率0%。')['result_yuan'],'0.00')

    def test_multiple_nulls_and_missing_paths_consistent(self):
        response=json.loads(read_split('train')[0]['messages'][2]['content'])
        response['data']['taxpayer_id']=None
        response['data']['items'][0]['amount_without_tax']=None
        response['data']['items'][0]['tax_rate']=None
        response['data']['missing_fields']=['taxpayer_id','items.0.amount_without_tax','items.0.tax_rate']
        validate_response(dumps(response))
        response['data']['missing_fields'].pop()
        with self.assertRaises(ValueError): validate_response(dumps(response))

    def test_blocked_wrong_call_still_counts_as_model_failure(self):
        gold={'action':'fallback','data':None,'reason':'insufficient_information'}
        guard=execute_checked(dumps(self.call),'不含税金额100元，税率未提供。')
        row={'id':'blocked','category':'fallback','scenario':'missing_calculation_rate','expected':gold,'raw_output':dumps(self.call),'latency_seconds':1,'runtime_guard':guard}
        self.assertEqual(guard['status'],'blocked')
        self.assertEqual(score([row])['fallback_exact']['correct'],0)
        self.assertEqual(boundary_score([row])['scenario_exact']['missing_calculation_rate']['correct'],0)

    def test_perfect_missing_predictions_and_malformed_denominator(self):
        row=next(r for r in read_split('test') if r['category']=='missing_extract')
        record={'id':row['id'],'category':row['category'],'expected':json.loads(row['messages'][2]['content']),'raw_output':row['messages'][2]['content'],'latency_seconds':1}
        bad={**record,'id':'malformed','raw_output':'not JSON'}
        result=boundary_score([record,bad])['missing_null_and_paths_exact']
        self.assertEqual((result['correct'],result['total']),(1,2))

if __name__=='__main__': unittest.main()
