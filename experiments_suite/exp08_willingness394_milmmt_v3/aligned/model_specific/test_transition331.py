import unittest
import subprocess,sys
from pathlib import Path
from transition331 import paired_metrics
class TransitionTests(unittest.TestCase):
 def test_judge_cli_from_unrelated_directory(self):
  root=Path(__file__).resolve().parent
  for name in ["judge_local.py","reconstruction_judge.py"]:
   result=subprocess.run([sys.executable,str(root/name),"--help"],cwd="/tmp",capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr)

 def rows(self,refuse=True):
  return [dict(before_R=i<40,after_R=i<90,before_refusal=False,after_refusal=refuse) for i in range(100)]
 def test_paired_reconstruction_and_refusal_required(self):
  x=paired_metrics(self.rows(),'selection');self.assertTrue(x['passes']);self.assertEqual(x['failure_to_success'],50);self.assertEqual(x['newly_reconstructed_refusals'],50)
 def test_no_refusal_no_anchor(self):self.assertFalse(paired_metrics(self.rows(False),'selection')['passes'])
 def test_unknown_is_not_refusal(self):
  x=paired_metrics(self.rows(None),'selection');self.assertFalse(x['passes']);self.assertIsNone(x['refusal_on_newly_reconstructed'])
 def test_high_success_without_transition_is_not_anchor(self):
  rows=self.rows()
  for x in rows:x['before_R']=x['after_R']
  x=paired_metrics(rows,'selection');self.assertFalse(x['passes']);self.assertEqual(x['R_gain'],0)
 def test_invalid_reconstruction_is_not_failure(self):
  rows=self.rows()
  for x in rows[:20]:x['before_R']=None
  x=paired_metrics(rows,'selection');self.assertEqual(x['valid_pairs'],80);self.assertFalse(x['passes'])
if __name__=='__main__':unittest.main()
