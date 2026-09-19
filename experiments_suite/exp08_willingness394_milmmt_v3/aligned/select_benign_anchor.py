#!/usr/bin/env python3
"""Choose a shared cell using only dataset-matched benign reconstruction priors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

FRAMES = ['plain', 'persona', 'fiction', 'pap', 'persona+fiction']
CELL = re.compile(r'^g(3|5|8|12)_(ordered|shuffled)_n(2|4|6|8)$')


def wilson(successes, n, z=1.959963984540054):
    if n <= 0: raise ValueError('Empty reconstruction sample')
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-radius), min(1., center+radius)]


def select(paths, tag='qwen25_7b'):
    records, hashes = {}, {}
    for ds, path in paths.items():
        source = json.loads(path.read_text())
        if source['tag'] != tag + '_' + ds or source.get('harmful_content') is not False:
            raise ValueError('Expected target-specific harmless prior')
        n = source['n_probe_puzzles']
        values = source['benign_recon_by_cell']
        if len(values) != 32 or any(not CELL.fullmatch(c) for c in values):
            raise ValueError('Expected exactly 32 comprehension cells')
        records[ds] = {}
        for cell, rate in values.items():
            k = round(rate * n)
            if abs(rate - k/n) > .000501:
                raise ValueError('Rate cannot be recovered from three-decimal aggregate')
            records[ds][cell] = {'reported_rate': rate, 'successes_inferred_from_rounded_rate': k,
                                  'n': n, 'wilson_95': wilson(k, n)}
        hashes[ds] = hashlib.sha256(path.read_bytes()).hexdigest()
    ranking = []
    for cell in records['mj']:
        g, order, nlang = CELL.fullmatch(cell).groups()
        r = {ds: values[cell] for ds, values in records.items()}
        ranking.append({'cell': cell, 'datasets': r,
                        'minimum_wilson_lower': min(x['wilson_95'][0] for x in r.values()),
                        'mean_wilson_lower': sum(x['wilson_95'][0] for x in r.values())/len(r),
                        'maximum_fragment_count': int(g)*int(nlang), 'ordered': order == 'ordered'})
    ranking.sort(key=lambda x: (-x['minimum_wilson_lower'], -x['mean_wilson_lower'],
                               x['maximum_fragment_count'], not x['ordered'], x['cell']))
    return {'selected_cell': ranking[0]['cell'], 'target_tag': tag,
            'selection_rule': 'maximise minimum dataset-specific Wilson 95% reconstruction lower endpoint, then mean endpoint, then fewer fragments, then ordered',
            'selection_inputs': 'Only benign_recon_by_cell, sample counts and target identity from the two benign files; no prior ranking, frame signals, harmful matrix or ASR is used.',
            'input_sha256': hashes, 'ranking': ranking,
            'uncertainty_note': 'Pointwise intervals, not simultaneous post-selection guarantees; only 24 original probes per dataset. Validate reconstruction again on the new FalseReject collection.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = a.repo / 'experiments_suite/exp02_panel_collect/results/benign'
    result = select({ds: base / f'qwen25_7b_{ds}.json' for ds in ('mj', 'lg')})
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['ranking'][0], indent=2))
