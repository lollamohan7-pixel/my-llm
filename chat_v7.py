import torch
import torch.nn.functional as F

from subword_tokenizer_v2 import tokenize
from gpt_v5_model import GPTv5


# ==========================================
# 1. LOAD GPT v7
# ==========================================

checkpoint = torch.load(
    "gpt_v7.pth",
    weights_only=True
)

vocab_size = checkpoint["vocab_size"]
vocabulary = checkpoint["vocabulary"]


# ==========================================
# 2. TOKEN MAPPINGS
# ==========================================

stoi = {
    token: i
    for i, token in enumerate(vocabulary)
}

itos = {
    i: token
    for i, token in enumerate(vocabulary)
}


# ==========================================
# 3. CREATE MODEL
# ==========================================

model = GPTv5(
    vocab_size=vocab_size,
    embedding_size=192,
    block_size=128,
    num_blocks=4
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================
# 4. SETTINGS
# ==========================================

TEMPERATURE = 0.7
MAX_NEW_TOKENS = 60


# ==========================================
# 5. DECODER
# ==========================================

def decode(tokens):

    result = ""

    for token in tokens:

        # New line
        if token == "\n":
            result += "\n"
            continue

        # Punctuation
        if token in [".", ",", "!", "?"]:

            result += token
            continue

        # New word marker
        if token.startswith("▁"):

            word = token[1:]

            if result and not result.endswith(
                (" ", "\n")
            ):
                result += " "

            result += word

        else:

            # Continuation of previous word
            result += token

    return result


# ==========================================
# 6. CHAT
# ==========================================

print("======================================")
print("             MY GPT v7")
print("======================================")
print("Type 'exit' to stop.")
print()


while True:

    question = input("You: ")

    if question.lower() == "exit":

        print("Goodbye!")

        break


    if not question.strip():
        continue


    # ======================================
    # CREATE PROMPT
    # ======================================

    prompt = (
        "User: "
        + question
        + "\nAssistant:"
    )


    # ======================================
    # TOKENIZE
    # ======================================

    prompt_tokens = tokenize(
        prompt
    )


    # ======================================
    # TOKEN → ID
    # ======================================

    token_ids = [

        stoi.get(
            token,
            stoi["<UNK>"]
        )

        for token in prompt_tokens
    ]


    context = torch.tensor(
        [token_ids],
        dtype=torch.long
    )


    # ======================================
    # GENERATE
    # ======================================

    with torch.no_grad():

        for _ in range(MAX_NEW_TOKENS):

            context_input = (
                context[:, -128:]
            )


            logits = model(
                context_input
            )


            # Last token prediction
            logits = logits[:, -1, :]


            # Temperature
            logits = (
                logits / TEMPERATURE
            )


            probabilities = F.softmax(
                logits,
                dim=-1
            )


            # Sample
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )


            context = torch.cat(
                [
                    context,
                    next_token
                ],
                dim=1
            )


            next_text = itos[
                next_token.item()
            ]


            # Stop at newline
            if next_text == "\n":

                break


    # ======================================
    # GET GENERATED TOKENS
    # ======================================

    all_tokens = [

        itos[i.item()]
        for i in context[0]
    ]


    generated_tokens = all_tokens[
        len(prompt_tokens):
    ]


    # ======================================
    # DECODE
    # ======================================

    answer = decode(
        generated_tokens
    )


    # ======================================
    # CLEAN RESPONSE
    # ======================================

    if "User:" in answer:

        answer = answer.split(
            "User:",
            1
        )[0]


    print()
    print(
        "GPT:",
        answer.strip()
    )
    print()