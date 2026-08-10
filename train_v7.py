import torch
import torch.nn.functional as F

from subword_tokenizer_v2 import tokenize
from gpt_v5_model import GPTv5


# ==========================================
# SETTINGS
# ==========================================

BLOCK_SIZE = 128
BATCH_SIZE = 16

EMBEDDING_SIZE = 192
NUM_BLOCKS = 4

STEPS = 5000
LEARNING_RATE = 0.0005


# ==========================================
# LOAD DATA
# ==========================================

with open(
    "chat_train.txt",
    "r",
    encoding="utf-8"
) as file:
    train_text = file.read()


with open(
    "chat_val.txt",
    "r",
    encoding="utf-8"
) as file:
    val_text = file.read()


# ==========================================
# TOKENIZE
# ==========================================

train_tokens = tokenize(train_text)
val_tokens = tokenize(val_text)


# ==========================================
# CREATE VOCABULARY
# ==========================================

vocabulary = sorted(
    set(train_tokens + val_tokens)
)

if "<UNK>" not in vocabulary:
    vocabulary.insert(0, "<UNK>")


stoi = {
    token: i
    for i, token in enumerate(vocabulary)
}

itos = {
    i: token
    for i, token in enumerate(vocabulary)
}


vocab_size = len(vocabulary)


print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))
print("Vocabulary size:", vocab_size)


# ==========================================
# TOKEN → IDs
# ==========================================

train_data = torch.tensor(
    [
        stoi.get(
            token,
            stoi["<UNK>"]
        )
        for token in train_tokens
    ],
    dtype=torch.long
)


val_data = torch.tensor(
    [
        stoi.get(
            token,
            stoi["<UNK>"]
        )
        for token in val_tokens
    ],
    dtype=torch.long
)


# ==========================================
# CREATE MODEL
# ==========================================

model = GPTv5(
    vocab_size=vocab_size,
    embedding_size=EMBEDDING_SIZE,
    block_size=BLOCK_SIZE,
    num_blocks=NUM_BLOCKS
)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# BATCH FUNCTION
# ==========================================

def get_batch(data):

    starts = torch.randint(
        0,
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack(
        [
            data[
                i:i + BLOCK_SIZE
            ]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:i + BLOCK_SIZE + 1
            ]
            for i in starts
        ]
    )

    return x, y


# ==========================================
# TRAIN
# ==========================================

for step in range(STEPS):

    x, y = get_batch(
        train_data
    )

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size
        ),
        y.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # ======================================
    # VALIDATION
    # ======================================

    if step % 500 == 0:

        with torch.no_grad():

            vx, vy = get_batch(
                val_data
            )

            val_logits = model(vx)

            val_loss = F.cross_entropy(
                val_logits.reshape(
                    -1,
                    vocab_size
                ),
                vy.reshape(-1)
            )


        print(
            f"Step {step} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Validation Loss: "
            f"{val_loss.item():.4f}"
        )


# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "vocab_size":
            vocab_size,

        "vocabulary":
            vocabulary
    },
    "gpt_v7.pth"
)


print()
print("Training complete!")
print("Model saved as gpt_v7.pth")