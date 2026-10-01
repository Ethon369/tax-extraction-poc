import copy
import json
import unittest
from src.common import read_split, dumps
from src.metrics import score
from src.schemas import validate_response
from src.tools import execute_checked

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.invoice=json.loads(read_split('train')[0]['messages'][2]['content'])
        self.tool={'action':'call_tool','data':{'tool_name':'calculate_tax_amount','parameters':{'amount_without_tax':0.015,'tax_rate':1}},'reason':None}

    def assertInvalid(self, response):
        with self.assertRaises(ValueError): validate_response(dumps(response))

    def test_all_labels(self):
        for split in ('train','dev','test','smoke'):
            for row in read_split(split): validate_response(row['messages'][2]['content'])

    def test_empty_object_rejected(self):
        self.assertInvalid({})

    def test_missing_amount_is_null_and_explicit(self):
        response=copy.deepcopy(self.invoice)
        response['data']['items'][0]['amount_without_tax']=None
        response['data']['missing_fields']=['items.0.amount_without_tax']
        validate_response(dumps(response))
        response['data']['missing_fields']=[]
        self.assertInvalid(response)

    def test_missing_taxpayer_id(self):
        response=copy.deepcopy(self.invoice)
        response['data']['taxpayer_id']=None
        response['data']['missing_fields']=['taxpayer_id']
        validate_response(dumps(response))

    def test_invalid_date(self):
        response=copy.deepcopy(self.invoice);response['data']['invoice_date']='2026-02-30'
        self.assertInvalid(response)

    def test_extra_fields(self):
        response=copy.deepcopy(self.invoice);response['data']['guessed_total']=123
        self.assertInvalid(response)

    def test_string_number_bool_and_nan(self):
        for value in ('123',True):
            response=copy.deepcopy(self.invoice);response['data']['items'][0]['amount_without_tax']=value
            self.assertInvalid(response)
        with self.assertRaises(ValueError): validate_response('{"action":"extract","data":NaN,"reason":null}')

    def test_duplicate_json_keys(self):
        with self.assertRaises(ValueError): validate_response('{"action":"fallback","action":"extract","data":null,"reason":"unsupported_request"}')

    def test_unknown_tool(self):
        response=copy.deepcopy(self.tool);response['data']['tool_name']='unregistered'
        self.assertEqual(execute_checked(dumps(response),'不含税金额0.015元，税率100%。')['status'],'blocked')

    def test_missing_tool_parameter(self):
        response=copy.deepcopy(self.tool);del response['data']['parameters']['tax_rate']
        self.assertEqual(execute_checked(dumps(response),'不含税金额0.015元')['status'],'blocked')

    def test_no_fabricated_or_ambiguous_parameters(self):
        for text in ('金额未说明，税率100%。','不含税金额0.015元或者不含税金额2元，税率100%。'):
            self.assertEqual(execute_checked(dumps(self.tool),text)['status'],'blocked')

    def test_decimal_rounding(self):
        result=execute_checked(dumps(self.tool),'不含税金额0.015元，税率100%。')
        self.assertEqual(result['status'],'executed');self.assertEqual(result['result_yuan'],'0.02')

    def test_deduction_ratio_must_be_explicit(self):
        response={'action':'call_tool','data':{'tool_name':'calculate_vat_deduction','parameters':{'tax_amount':75,'deductible_ratio':.4}},'reason':None}
        self.assertEqual(execute_checked(dumps(response),'税额75元，可抵扣比例40%。')['result_yuan'],'30.00')
        self.assertEqual(execute_checked(dumps(response),'税额75元，可抵扣资格未知。')['status'],'blocked')

    def test_perfect_metrics(self):
        records=[]
        for row in read_split('test'):
            answer=json.loads(row['messages'][2]['content'])
            records.append({'id':row['id'],'expected':answer,'raw_output':dumps(answer),'latency_seconds':1})
        result=score(records)
        self.assertEqual(result['schema_valid']['rate'],1)
        self.assertEqual(result['field_accuracy']['rate'],1)
        self.assertEqual(result['non_null_field_micro_f1']['f1'],1)
        self.assertEqual(result['tool_exact']['rate'],1)

    def test_parse_failure_stays_in_denominators(self):
        rows=[{'id':'broken','expected':self.invoice,'raw_output':'not JSON','latency_seconds':1}]
        result=score(rows)
        self.assertEqual(result['json_parse']['total'],1)
        self.assertEqual(result['field_accuracy']['correct'],0)
        self.assertGreater(result['non_null_field_micro_f1']['fn'],0)

    def test_hallucinated_null_field_counted(self):
        gold=copy.deepcopy(self.invoice);gold['data']['taxpayer_id']=None;gold['data']['missing_fields']=['taxpayer_id']
        result=score([{'id':'invented','expected':gold,'raw_output':dumps(self.invoice),'latency_seconds':1}])
        self.assertEqual(result['invented_values_on_null_fields'],1)

if __name__=='__main__': unittest.main()
