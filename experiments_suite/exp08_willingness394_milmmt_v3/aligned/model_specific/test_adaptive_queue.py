import unittest
import tempfile,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
from adaptive_queue import bend,candidates,gpu_budget
class CurveTests(unittest.TestCase):
 def test_gpu_budget_reserves_headroom(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':14*1024**3}}))
   with patch('adaptive_queue.subprocess.check_output',return_value='81000, 50000'):
    x=gpu_budget(p);self.assertTrue(x['ready']);self.assertLess(x['utilization'],.60)
   with patch('adaptive_queue.subprocess.check_output',return_value='81000, 18000'):
    self.assertFalse(gpu_budget(p)['ready'])
 def test_shared_gpu_judge_cli(self):
  p=Path(__file__).resolve().with_name('queue_judge.py')
  r=subprocess.run([sys.executable,str(p),'--help'],cwd='/tmp',capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)

 def obs(self,rates):
  return {f'g{g}_{o}_n2':{str(i):dict(R=i<int(rate*100+.01),F=True) for i in range(100)} for o in ['ordered','shuffled'] for g,rate in zip([12,8,5,3],rates)}
 def test_clear_knee(self):
  ranking=candidates(self.obs([.10,.85,.90,.94]),2)
  self.assertTrue(ranking[0]['passes']);self.assertEqual(ranking[0]['after'].split('_')[0],'g8')
 def test_linear_curve_is_not_a_knee(self):
  x=bend(self.obs([.40,.60,.75,.85]),['g12_ordered_n2','g8_ordered_n2','g5_ordered_n2'])
  self.assertAlmostEqual(x['change'],0);self.assertFalse(x['nonzero'])
 def test_knee_without_refusal_is_not_selected(self):
  obs=self.obs([.10,.85,.90,.94])
  for values in obs.values():
   for x in values.values():x['F']=False
  self.assertFalse(any(x['passes'] for x in candidates(obs,2)))
if __name__=='__main__':unittest.main()
