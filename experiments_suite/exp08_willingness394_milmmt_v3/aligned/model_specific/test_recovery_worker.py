import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import adaptive_queue_worker as worker

class RecoveryWorkerTests(unittest.TestCase):
 def test_selected_gpu_controls_memory_probe(self):
  previous=worker.GPU
  try:
   worker.GPU=1
   with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':14*1024**3}}))
    with patch.object(worker.subprocess,'check_output',return_value='81559, 80158') as check:
     self.assertTrue(worker.gpu_budget(p)['ready'])
     self.assertIn('--id=1',check.call_args.args[0])
  finally:worker.GPU=previous
if __name__=='__main__':unittest.main()
