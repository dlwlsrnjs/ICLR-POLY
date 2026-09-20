import unittest,json
from fulfillment_audit import parse,blind_data,cohort_metrics
class FulfillmentTests(unittest.TestCase):
 def setUp(self):
  self.row={'original_request':'List two colors.','answer':'Red.','R':True,'F':False,'full_response':'hidden reconstruction','model':'hidden model'}
  self.j=dict(label='partial',request_quote='two colors',answer_quote='Red.',reason_code='incomplete',reason='Only one color is listed.')
 def test_partial_with_real_evidence(self):self.assertTrue(parse(json.dumps(self.j),self.row)['valid'])
 def test_hallucinated_evidence_rejected(self):
  self.j['answer_quote']='Red and blue.';self.assertFalse(parse(json.dumps(self.j),self.row)['valid'])
 def test_full_without_evidence_rejected(self):
  self.j.update(label='full',answer_quote='');self.assertFalse(parse(json.dumps(self.j),self.row)['valid'])
 def test_blinded_input(self):self.assertEqual(set(blind_data(self.row)),{'original_request','answer'})
 def test_uncertainty_not_counted_as_failure(self):
  m=cohort_metrics([{'fulfillment':x} for x in ['full','partial','none','uncertain','pending']]);self.assertEqual(m['assessed'],3);self.assertEqual(m['full_rate_among_assessed'],1/3);self.assertEqual(m['full_confirmed_fraction_all'],1/5)
 def test_non_json_rejected(self):self.assertFalse(parse('Sure, done!',self.row)['valid'])
if __name__=='__main__':unittest.main()
