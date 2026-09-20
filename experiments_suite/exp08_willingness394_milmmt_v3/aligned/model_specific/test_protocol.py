import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import calibrate
from analyze import complexity,wilson

class ProtocolTests(unittest.TestCase):
 def test_language_profiles(self):
  self.assertEqual(calibrate.profile('g3_ordered_n2'),['English','Arabic'])
  self.assertEqual(calibrate.profile('g3_ordered_n4'),['English','Arabic','Chinese','Norwegian'])
  self.assertEqual(len(calibrate.profile('g3_ordered_n8')),8)
 def test_simple_tie_break(self):
  self.assertLess(complexity('g3_ordered_n2'),complexity('g3_ordered_n8'))
  self.assertLess(complexity('g3_ordered_n2'),complexity('g3_shuffled_n2'))
 def test_small_sample_uncertainty(self):
  self.assertLess(wilson(24,24)[0],.90)
  self.assertLess(wilson(45,50)[0],.80)
 def test_freeze_does_not_select_on_validation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'model_specific';root.mkdir();run=Path(tmp)/'run';(run/'reconstruction').mkdir(parents=True)
   cfg=root.parent/'shared_en_ar/configs';cfg.mkdir(parents=True);(cfg/'m.json').write_text('{}')
   cells=['g3_ordered_n2','g3_ordered_n4']
   (run/'calibration_plan.json').write_text(json.dumps({'models':[{'model':'m','jobs':200,'cells':cells}]}))
   rows=[]
   for cell in cells:
    for split in ['selection','validation']:
     # n4 wins selection 48 versus 40; n2 wins validation 50 versus 20.
     k={('g3_ordered_n2','selection'):40,('g3_ordered_n4','selection'):48,('g3_ordered_n2','validation'):50,('g3_ordered_n4','validation'):20}[cell,split]
     for j in range(50):rows.append(dict(key=f'{cell}:{split}:{j}',model_tag='m',condition=cell,split=split,finish_reason='stop',section_valid=True,reconstruction_parse_valid=1,semantic_reconstruction_equivalent=int(j<k)))
   inp=[{k:v for k,v in x.items() if k not in ('reconstruction_parse_valid','semantic_reconstruction_equivalent')} for x in rows]
   path=run/'reconstruction/restricted_reconstruction_audit.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   (run/'reconstruction_inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in inp))
   calibrate.freeze(root,run);out=json.loads((run/'anchor_decisions.json').read_text())['models'][0]
   self.assertEqual(out['cell'],'g3_ordered_n4');self.assertFalse(out['validation_adequate'])
   rows[0]['reconstruction_parse_valid']=0;path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
   calibrate.freeze(root,run);out=json.loads((run/'anchor_decisions.json').read_text())['models'][0]
   self.assertEqual(out['status'],'blocked_incomplete_or_invalid_judgments')
if __name__=='__main__':unittest.main()
