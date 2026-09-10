"""Swap the low-resource-answer role-separation arms for their English-answer counterparts in both
bandit loaders. Run only after mj_hien / lingua_hien results exist for all nine targets."""
import sys

PATCHES = {
    "scripts/mj_bandit_full.py": (
        '''    tri = jload(f"{R}/mj_triple_20260906/{t}.json")["variants"]     # our strengthened arms
    tf = {"hi_wl": [0.4, 0, 0, 0, 1], "triple": [0.4, 0, 1, 0, 1], "triple_en": [0.4, 0, 1, 0, 0]}
    for k, v in tri.items():
        out.append(_row(f"tri_{k}", v, tf[k]))''',
        '''    # role separation, English answer only. Low-resource answers are dropped: the safety judge
    # scores an English answer reliably but over-counts a low-resource one (Section on judge
    # accuracy). hi_en is the persona-free arm, triple_en the persona arm; both answer in English.
    tri = jload(f"{R}/mj_triple_20260906/{t}.json")["variants"]
    hien = jload(f"{R}/mj_hien_20260908/{t}.json")["variants"]
    out.append(_row("tri_hi_en", hien["hi_en"], [0.4, 0, 0, 0, 0]))
    out.append(_row("tri_triple_en", tri["triple_en"], [0.4, 0, 1, 0, 0]))''',
    ),
    "scripts/lingua_bandit_full.py": (
        '''    tri = jload(f"{R}/lingua_triple_20260907/{t}.json")["variants"]   # role-separation arm (Norwegian answer)
    tf = {"hi_wl": [0.4, 0, 0, 1], "triple": [0.4, 0, 1, 1], "triple_en": [0.4, 0, 1, 0]}
    for k, v in tri.items():
        out.append(_row(f"tri_{k}", v, tf[k]))''',
        '''    # role separation, English answer only (see Section on judge accuracy).
    tri = jload(f"{R}/lingua_triple_20260907/{t}.json")["variants"]
    hien = jload(f"{R}/lingua_hien_20260908/{t}.json")["variants"]
    out.append(_row("tri_hi_en", hien["hi_en"], [0.4, 0, 0, 0]))
    out.append(_row("tri_triple_en", tri["triple_en"], [0.4, 0, 1, 0]))''',
    ),
}

for f, (old, new) in PATCHES.items():
    s = open(f).read()
    if new in s:
        print(f, "already patched"); continue
    assert s.count(old) == 1, f"anchor not unique in {f}"
    open(f, "w").write(s.replace(old, new))
    print(f, "patched to English-answer arms")
