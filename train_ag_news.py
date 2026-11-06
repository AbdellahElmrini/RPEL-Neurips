import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from src import misc

# ------------------------
# Config
# ------------------------
#DATA_DIR = "./data/ag_news"

DATA_DIR = misc.get_default_root() / "ag_news"
TOKENIZER_PATH = str(misc.get_default_root() / "ag_news_tokenizer.json")

BATCH_SIZE = 64
EMBED_DIM = 200
EPOCHS = 15
LR = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load dataset (cached)
# ------------------------
dataset = load_dataset("ag_news", cache_dir=DATA_DIR)

# ------------------------
# Build or load tokenizer
# ------------------------
if os.path.exists(TOKENIZER_PATH):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
else:
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        vocab_size=MAX_VOCAB,
        min_frequency=MIN_FREQ,
        special_tokens=["<unk>", "<pad>"]
    )

    tokenizer.train_from_iterator(
        (x["text"] for x in dataset["train"]),
        trainer
    )

    tokenizer.save(TOKENIZER_PATH)

PAD_ID = tokenizer.token_to_id("<pad>")

# ------------------------
# Collate function
# ------------------------
def collate_fn(batch):
    texts = [
        torch.tensor(tokenizer.encode(x["text"]).ids, dtype=torch.long)
        for x in batch
    ]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    texts = pad_sequence(
        texts, batch_first=True, padding_value=PAD_ID
    )
    return texts.to(DEVICE), labels.to(DEVICE)

# ------------------------
# DataLoaders
# ------------------------
train_loader = DataLoader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    dataset["test"],
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

# ------------------------
# Model (FastText-style)
# ------------------------
class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=PAD_ID
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)               # (B, T, D)
        mask = (x != PAD_ID).unsqueeze(-1)    # (B, T, 1)
        emb = emb * mask

        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(pooled)

model = FastTextClassifier(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=EMBED_DIM,
    num_classes=4
).to(DEVICE)

# ------------------------
# Training setup
# ------------------------
#optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=0.9,
    weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()
# ------------------------
# Train / Eval loops
# ------------------------
def train_epoch():
    model.train()
    total_loss = 0

    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0

    for x, y in test_loader:
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total

# ------------------------
# Run training
# ------------------------

acc = evaluate()
print(f"Epoch {0}: test_acc={acc:.4f}")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch()
    acc = evaluate()
    print(f"Epoch {epoch}: loss={loss:.4f}, test_acc={acc:.4f}")
