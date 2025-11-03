from pathlib import Path
import pyprind
import pandas as pd

base = Path(r"C:\Users\porsche\Documents\GitHub\Today-I-Learned\aclImdb_v1") / "aclImdb"
label_map = {'pos': 1, 'neg': 0}

# 필수 디렉터리 확인
for must in [base, base/"train"/"pos", base/"test"/"neg"]:
    if not must.exists():
        raise FileNotFoundError(f"폴더가 없습니다: {must}")

# 실제 txt 개수 기반 프로그레스바
total = sum(1 for s in ("train","test") for l in ("pos","neg") for _ in (base/s/l).glob("*.txt"))
pbar = pyprind.ProgBar(total if total else 50000)

rows = []
for s in ("test", "train"):
    for l in ("pos", "neg"):
        d = base / s / l
        for fp in sorted(d.glob("*.txt")):
            try:
                txt = fp.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                txt = fp.read_text(encoding="latin-1")
            rows.append([txt, label_map[l]])
            pbar.update()

df = pd.DataFrame(rows, columns=["review", "sentiment"])
