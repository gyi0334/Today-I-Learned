# ===== 1) 데이터 =====
texts = [
    "This movie was fantastic! I loved it.",
    "Absolutely terrible. Worst film ever.",
    "Great performance and wonderful story.",
    "I hated the movie, it was boring.",
]
labels = [1, 0, 1, 0]  # 1 = 긍정, 0 = 부정

# ===== 2) 전처리/어휘 =====
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

def tokenize(text):
    return text.lower().replace("!", "").replace(".", "").split()

# 동적 구축 단계에서는 defaultdict 사용
word2idx = defaultdict(lambda: len(word2idx))
PAD_IDX = word2idx["<PAD>"]
UNK_IDX = word2idx["<UNK>"]   # OOV 토큰 미리 추가

tokenized = [tokenize(t) for t in texts]
encoded = [[word2idx[token] for token in seq] for seq in tokenized]  # 훈련 텍스트는 여기서 어휘 확장

# 어휘 고정(Freeze): 이후에는 더 이상 커지지 않도록 dict로 변환
word2idx = dict(word2idx)
vocab_size = len(word2idx)

# 텐서/패딩
tensor_seqs = [torch.tensor([word2idx.get(tok, UNK_IDX) for tok in seq], dtype=torch.long)
               for seq in tokenized]
padded = pad_sequence(tensor_seqs, batch_first=True, padding_value=PAD_IDX)
labels_tensor = torch.tensor(labels, dtype=torch.long)

print("Vocabulary size:", vocab_size)

# ===== 3) 모델 =====
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.pad_idx = pad_idx

    def forward(self, x):
        embedded = self.embedding(x)                         # (B, L, D)
        mask = (x != self.pad_idx).unsqueeze(-1)            # (B, L, 1)
        summed = (embedded * mask).sum(1)                   # (B, D)
        counts = mask.sum(1).clamp(min=1)                   # (B, 1)
        avg = summed / counts                               
        return self.fc(avg)                                 # (B, C)

model = TextClassifier(vocab_size=vocab_size, embed_dim=16, num_classes=2, pad_idx=PAD_IDX)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ===== 4) 학습 =====
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(padded)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ===== 5) 추론 =====
def predict(text: str):
    model.eval()
    tokens = [word2idx.get(tok, UNK_IDX) for tok in tokenize(text)]  # OOV는 UNK로
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)          # (1, L)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    return ("긍정" if pred == 1 else "부정"), probs.squeeze(0).tolist()

print(predict("I really love this amazing movie"))     # OOV 단어 포함해도 안전
print(predict("This was the worst experience ever"))
