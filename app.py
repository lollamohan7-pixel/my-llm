from flask import Flask, request, render_template_string
import os
import torch
import torch.nn.functional as F

from gpt_v5_model import GPTv5
from subword_tokenizer_v2 import tokenize


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# MODEL FILE PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "gpt_v7.pth"
)


# ==========================================
# LOAD MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

vocab_size = checkpoint["vocab_size"]
vocabulary = checkpoint["vocabulary"]


# ==========================================
# TOKEN → ID
# ==========================================

stoi = {
    token: i
    for i, token in enumerate(vocabulary)
}


# ==========================================
# ID → TOKEN
# ==========================================

itos = {
    i: token
    for i, token in enumerate(vocabulary)
}


# ==========================================
# CREATE MODEL
# ==========================================

model = GPTv5(
    vocab_size=vocab_size,
    embedding_size=192,
    block_size=128,
    num_blocks=4
)


# ==========================================
# LOAD TRAINED WEIGHTS
# ==========================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ==========================================
# EVALUATION MODE
# ==========================================

model.eval()


# ==========================================
# TEXT DECODER
# ==========================================

def decode(tokens):

    result = ""

    for token in tokens:

        # New line
        if token == "\n":
            result += "\n"
            continue

        # Punctuation
        if token in [".", ",", "!", "?", ":"]:
            result += token
            continue

        # Word starts with ▁
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

    # Create prompt
    prompt = (
        "User: "
        + question
        + "\nAssistant:"
    )


    # ======================================
    # TOKENIZE
    # ======================================

    prompt_tokens = tokenize(prompt)


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


    # ======================================
    # CREATE TENSOR
    # ======================================

    context = torch.tensor(
        [token_ids],
        dtype=torch.long
    )


    # ======================================
    # GENERATION
    # ======================================

    with torch.inference_mode():

        # Generate maximum 20 tokens
        for _ in range(20):

            # Keep latest 128 tokens
            context_input = context[:, -128:]


            # Model prediction
            logits = model(
                context_input
            )


            # Get last token prediction
            logits = logits[:, -1, :]


            # Temperature
            logits = logits / 0.7


            # Convert logits to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )


            # Choose most likely token
            next_token = torch.argmax(
                probabilities,
                dim=-1,
                keepdim=True
            )


            # Add token to context
            context = torch.cat(
                [
                    context,
                    next_token
                ],
                dim=1
            )


            # Convert ID → token
            next_text = itos[
                next_token.item()
            ]


            # Stop at new line
            if next_text == "\n":
                break


    # ======================================
    # GET ALL TOKENS
    # ======================================

    all_tokens = [
        itos[i.item()]
        for i in context[0]
    ]


    # Remove prompt tokens
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
    # REMOVE EXTRA USER PROMPT
    # ======================================

    if "User:" in answer:

        answer = answer.split(
            "User:",
            1
        )[0]


    return answer.strip()


# ==========================================
# HTML PAGE
# ==========================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>My GPT v7</title>

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
# HOME ROUTE
# ==========================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

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