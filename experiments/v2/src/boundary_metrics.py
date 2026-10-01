"""Additional metrics score raw model predictions, never runtime guard repairs."""
from .schemas import strict_json_loads, validate_response
from .metrics import equal_nested

def boundary_score(records):
    groups = {}
    for row in records:
        expected = row['expected']
        try:
            prediction = strict_json_loads(row['raw_output'])
            valid = True
            try:
                validate_response(row['raw_output'])
            except (ValueError, TypeError):
                valid = False
        except (ValueError, TypeError):
            prediction, valid = {}, False
        if not isinstance(prediction, dict):
            prediction = {}
        name = row.get('scenario') or row['category']
        group = groups.setdefault(name, {'correct':0,'total':0})
        group['total'] += 1
        group['correct'] += int(valid and equal_nested(prediction, expected))
    missing = [r for r in records if r['category']=='missing_extract']
    correct = 0
    for row in missing:
        try:
            pred = strict_json_loads(row['raw_output'])
            validate_response(row['raw_output'])
            gold = row['expected']['data']
            if pred['action']=='extract':
                from .metrics import flatten
                expected_flat, actual_flat = flatten(gold), flatten(pred['data'])
                correct += int(set(pred['data']['missing_fields']) == set(gold['missing_fields']) and all(path in actual_flat and actual_flat[path] is None for path,value in expected_flat.items() if value is None))
        except (ValueError,TypeError,KeyError):
            pass
    return {'missing_null_and_paths_exact':{'correct':correct,'total':len(missing),'rate':correct/len(missing) if missing else None}, 'scenario_exact':groups, 'raw_prediction_only':True}
