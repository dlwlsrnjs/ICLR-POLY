"""Load the repository's exact grid renderer without importing GPU dependencies.

Only named constant assignments and _compose from the trusted repository source
are compiled. Prompts/config data are never executed. All source hashes are pinned
by the run config; changes require a new config/run.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

FRAMES = ['plain', 'persona', 'fiction', 'pap', 'persona+fiction']
SOURCE_FILES = ['scripts/closed_compare.py', 'scripts/combo_eval.py',
                'scripts/run_qwen_interleaving_probe.py', 'scripts/online_live.py']


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_hashes(repo):
    return {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def literals(path, names):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != set(names): raise ValueError('Repository constant contract changed')
    return result


class GridContract:
    def __init__(self, repo, expected_hashes):
        self.repo = Path(repo)
        self.hashes = source_hashes(self.repo)
        if self.hashes != expected_hashes:
            raise ValueError('Grid source changed; audit and freeze a new config rather than silently mixing protocols')
        path = self.repo/'scripts/run_qwen_interleaving_probe.py'
        spec = importlib.util.spec_from_file_location('aligned_original_puzzle', path)
        self.puzzle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.puzzle)
        seed = literals(self.repo/'scripts/combo_eval.py', ['SEED'])['SEED']
        self.seed = seed
        source = self.repo/'scripts/closed_compare.py'
        ns = literals(source, ['AIM_HEAD', 'FICTION_HEAD', 'PAP_HEAD', 'REASSEMBLE', 'C_WILL'])
        if set(FRAMES) != ns['C_WILL']: raise ValueError('Five-frame contract changed')
        ns.update(FRAME_HEAD={'persona': ns['AIM_HEAD'], 'fiction': ns['FICTION_HEAD'], 'pap': ns['PAP_HEAD']},
                  os=SimpleNamespace(environ={}), CE=SimpleNamespace(mixed=self.mixed))
        node = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.FunctionDef) and x.name == '_compose')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
        self.compose = ns['_compose']

    def mixed(self, row, languages, n, fragments, order):
        p = self.puzzle.build_puzzle(row, languages[:n], order, self.seed, 'coarse', fragments)
        return p['mixed_paragraph'], languages[:n]

    def render(self, row, languages, fragments, order, frame):
        if frame not in FRAMES: raise ValueError('Only the five C-space frames are supported')
        if order not in ('ordered', 'shuffled'): raise ValueError('Order outside C space')
        if any(not str(row['questions'].get(lang, '')).strip() for lang in languages):
            raise ValueError('Missing language; no English or other-language fallback')
        p = self.puzzle.build_puzzle(row, languages, order, self.seed, 'coarse', fragments)
        ws = () if frame == 'plain' else tuple(frame.split('+'))
        prompt = self.compose(row, languages, len(languages), fragments, order, ws)
        return {'prompt': prompt, 'messages': [{'role': 'user', 'content': prompt}],
                'payload_sha256': hashlib.sha256(p['mixed_paragraph'].encode()).hexdigest(),
                'actual_fragments': len(p['records']),
                'fragment_records': p['records'], 'gold_ids': p['gold_ids'],
                'gold_english': p['gold_english'], 'source_translations': row['questions']}
