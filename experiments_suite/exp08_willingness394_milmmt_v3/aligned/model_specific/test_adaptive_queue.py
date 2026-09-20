import unittest
from adaptive_queue import bend,candidates
class CurveTests(unittest.TestCase):
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
