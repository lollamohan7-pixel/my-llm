from flask import Flask, request, render_template_string
import torch
import torch.nn.functional as F

from gpt_v5_model import GPTv5
from subword_tokenizer_v2 import tokenize


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD MODEL
# ==========================================

checkpoint = torch.load(
    "gpt_v7.pth",
    map_location="cpu",
    weights_only=True
)

vocab_size = checkpoint["vocab_size"]
vocabulary = checkpoint["vocabulary"]


stoi = {
    token: i
    for i, token in enumerate(vocabulary)
}

itos = {
    i: token
    for i, token in enumerate(vocabulary)
}


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
# TEXT DECODER
# ==========================================

def decode(tokens):

    result = ""

    for token in tokens:

        if token == "\n":
            result += "\n"
            continue

        if token in [".", ",", "!", "?", ":"]:
            result += token
            continue

        if token.startswith("▁"):

            word = token[1:]

            if result and not result.endswith(
                (" ", "\n")
            ):
                result += " "

            result += word

        else:

            result += token

    return result


# ==========================================
# GENERATE ANSWER
# ==========================================

def generate_answer(question):

    prompt = (
        "User: "
        + question
        + "\nAssistant:"
    )

    prompt_tokens = tokenize(prompt)

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

    with torch.no_grad():

        for _ in range(80):

            context_input = context[:, -128:]

            logits = model(context_input)

            logits = logits[:, -1, :]

            logits = logits / 0.7

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                1
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

            if next_text == "\n":
                break

    all_tokens = [
        itos[i.item()]
        for i in context[0]
    ]

    generated_tokens = all_tokens[
        len(prompt_tokens):
    ]

    answer = decode(
        generated_tokens
    )

    if "User:" in answer:
        answer = answer.split(
            "User:",
            1
        )[0]

    return answer.strip()


# ==========================================
# WEB PAGE
# ==========================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>My GPT</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    max-width: 800px;
    margin: 50px auto;
    padding: 20px;
}

h1 {
    text-align: center;
}

textarea {
    width: 100%;
    height: 100px;
    padding: 15px;
    font-size: 16px;
    box-sizing: border-box;
    border-radius: 10px;
}

button {
    margin-top: 15px;
    padding: 12px 25px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 8px;
}

.answer {
    margin-top: 30px;
    padding: 20px;
    background: #222;
    border-radius: 10px;
    white-space: pre-wrap;
}

</style>

</head>

<body>

<h1>🤖 My GPT v7</h1>

<form method="POST">

<textarea
name="question"
placeholder="Ask something..."
required
>{{ question }}</textarea>

<br>

<button type="submit">
Ask GPT
</button>

</form>

{% if answer %}

<div class="answer">

<strong>GPT:</strong>

<br><br>

{{ answer }}

</div>

{% endif %}

</body>

</html>
"""


# ==========================================
# ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        )

        if question.strip():

            answer = generate_answer(
                question
            )

    return render_template_string(
        HTML,
        question=question,
        answer=answer
    )


# ==========================================
# RUN LOCALLY
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )