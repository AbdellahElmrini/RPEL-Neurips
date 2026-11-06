import time
start = time.time()
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

print("Imports took {:.2f} seconds".format(time.time() - start))
start2 = time.time()
# ------------------------
# Config
# ------------------------
#DATA_DIR = "./data/ag_news"

DATA_DIR = misc.get_default_root() / "ag_news"
TOKENIZER_PATH = str(misc.get_default_root() / "ag_news_tokenizer.json")

BATCH_SIZE = 64
EMBED_DIM = 200
NUM_ROUNDS = 500
EVAL_EVERY = 5
LR = 0.5
N = 5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load dataset (cached)
# ------------------------

dataset = load_dataset("ag_news", cache_dir=DATA_DIR)

print("Loading dataset took {:.2f} seconds".format(time.time() - start2))
# Split the dataset on N clients for federated learning in random fashion

def split_dataset(dataset, n):
    dataset = dataset.shuffle(seed=42)
    data_splits = []
    total_size = len(dataset)
    split_size = total_size // n
    for i in range(n):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != n - 1 else total_size
        data_splits.append(dataset.select(range(start_idx, end_idx)))
    return data_splits


data_splits = split_dataset(dataset["train"], N)

print(len(dataset["train"]))

print(len(data_splits[0]))

#raise SystemExit("Stop after dataset split for federated learning")

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

train_loaders = []
for i in range(N):
    train_loaders.append(DataLoader(
        data_splits[i],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )) 

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

# Initialize local models for each client

local_models = []
for i in range(N):
    local_model = FastTextClassifier(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=EMBED_DIM,
        num_classes=4
    ).to(DEVICE)
    local_models.append(local_model)


# Initialize local optimizers for each client
local_optimizers = []

for i in range(N):
    local_optimizer = torch.optim.SGD(
        local_models[i].parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=1e-4
    )
    local_optimizers.append(local_optimizer)
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
def local_training_step():
    model.train()
    total_loss = 0
    
    losses = []
    for i in range(N):
        local_models[i].load_state_dict(model.state_dict())
        local_optimizers[i].zero_grad()
        x,y = next(iter(train_loaders[i]))
        logits = local_models[i](x)
        loss = criterion(logits, y)
        loss.backward()
        local_optimizers[i].step()
        total_loss += loss.item()
        losses.append(loss.item())


    return losses # / len(train_loader)

def aggregate_models():
    global_dict = model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [local_models[i].state_dict()[k].float() for i in range(N)],
            0
        ).mean(0)
    model.load_state_dict(global_dict)

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
start3 = time.time()
acc = evaluate()
print(f"Epoch {0}: test_acc={acc:.4f}")
for ROUND in range(1, NUM_ROUNDS + 1):
    losses = local_training_step()
    aggregate_models()
    loss= sum(losses)/len(losses)
    if ROUND % EVAL_EVERY == 0:
        acc = evaluate()
        print(f"Round {ROUND}: loss={loss}, test_acc={acc:.4f}")

print("Total training took {:.2f} seconds".format(time.time() - start3))