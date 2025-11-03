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


import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))
print(df.shape)


# 문서를 토큰으로 나누기
import nltk
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
if w not in stop]

# 문서 분류를 위한 로지스틱 회귀 모델 훈련
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

print('최적의 매개변수 조합: %s ' % gs_lr_tfidf.best_params_)
print('CV 정확도: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('테스트 정확도: %.3f' % clf.score(X_test, y_test))