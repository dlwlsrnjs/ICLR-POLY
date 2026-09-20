import unittest
from matched_n2_report import validate_jobs
class MatchedTests(unittest.TestCase):
 def setUp(self):
  self.contract={'cells':['g3_ordered_n2','g5_ordered_n2'],'selection_item_ids':['a','b']}
  self.jobs=[dict(cell=c,id=i,frame='plain') for c in self.contract['cells'] for i in ['a','b']]
 def test_identical_cohort(self):self.assertTrue(validate_jobs(self.jobs,self.contract))
 def test_n6_excluded(self):
  for r in self.jobs:r['cell']=r['cell'].replace('n2','n6')
  self.assertFalse(validate_jobs(self.jobs,self.contract))
 def test_equal_count_different_item_rejected(self):
  self.jobs[0]['id']='c'
  with self.assertRaises(ValueError):validate_jobs(self.jobs,self.contract)
if __name__=='__main__':unittest.main()
