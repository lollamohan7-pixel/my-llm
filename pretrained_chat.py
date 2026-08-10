import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# MODEL
# ==========================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading pretrained model...")
print("The first run will download the model.")
print()


# ==========================================
# LOAD TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# ==========================================
# LOAD MODEL
# ==========================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
)

model.eval()


print("Model loaded successfully!")
print("Type 'exit' to stop.")
print()


# ==========================================
# CHAT
# ==========================================

while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    # Convert conversation into model input
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Generate answer
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )

    # Only keep newly generated tokens
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    # Convert tokens back to text
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print()
    print("Qwen:", answer)
    print()