import random


# ==========================================
# 1. Question and answer examples
# ==========================================

conversations = [

    """
User: What is Python?
Assistant: Python is a high level programming language used for software development, automation, data science, and artificial intelligence.

""",

    """
User: What is artificial intelligence?
Assistant: Artificial intelligence is a field of computer science that allows machines to perform tasks that normally require human intelligence.

""",

    """
User: What is machine learning?
Assistant: Machine learning is a method where computers learn patterns from data and use those patterns to make predictions or decisions.

""",

    """
User: What is a neural network?
Assistant: A neural network is a mathematical model made of connected layers that learns patterns from data.

""",

    """
User: What is deep learning?
Assistant: Deep learning is a type of machine learning that uses neural networks with multiple layers.

""",

    """
User: What is a transformer?
Assistant: A transformer is a neural network architecture that uses attention to understand relationships between tokens in a sequence.

""",

    """
User: What is self attention?
Assistant: Self attention allows a model to determine which parts of a sequence are important when processing each token.

""",

    """
User: What is a language model?
Assistant: A language model learns patterns in text and predicts the next token in a sequence.

""",

    """
User: What is a tokenizer?
Assistant: A tokenizer converts text into smaller units called tokens so that a neural network can process the text.

""",

    """
User: What is an algorithm?
Assistant: An algorithm is a sequence of steps used to solve a problem.

""",

    """
User: What is a data structure?
Assistant: A data structure is a way of organizing and storing data so that it can be used efficiently.

""",

    """
User: What is an array?
Assistant: An array stores multiple elements in an ordered sequence.

""",

    """
User: What is a stack?
Assistant: A stack is a data structure that follows the last in first out principle.

""",

    """
User: What is a queue?
Assistant: A queue is a data structure that follows the first in first out principle.

""",

    """
User: What is a computer?
Assistant: A computer is an electronic machine that processes instructions and data to perform useful tasks.

""",

    """
User: What is a processor?
Assistant: A processor is a hardware component that executes instructions and performs calculations.

""",

    """
User: What is memory?
Assistant: Computer memory stores data and instructions that programs need while they are running.

""",

    """
User: What is electronics?
Assistant: Electronics is the field of engineering concerned with controlling electrical signals using electronic components.

""",

    """
User: What is a transistor?
Assistant: A transistor is an electronic component that can work as a switch or amplifier.

""",

    """
User: What is a microcontroller?
Assistant: A microcontroller is a small computer on a chip containing a processor, memory, and input output interfaces.

""",

    """
User: What is IoT?
Assistant: The Internet of Things connects physical devices to networks so that they can collect, exchange, and process information.

""",

    """
User: What is programming?
Assistant: Programming is the process of writing instructions that a computer can execute.

""",

    """
User: Why is Python popular?
Assistant: Python is popular because its syntax is relatively simple and it has many libraries for different applications.

""",

    """
User: Why should I learn programming?
Assistant: Learning programming develops problem solving skills and allows you to create software and automate tasks.

""",

    """
User: How does machine learning work?
Assistant: Machine learning works by training a model on data so that it can learn patterns and make predictions on new data.

""",

    """
User: How does a neural network learn?
Assistant: A neural network learns by comparing its predictions with the correct answers and adjusting its parameters to reduce the error.

""",

    """
User: How does a language model generate text?
Assistant: A language model predicts the next token based on the tokens that came before it and repeats this process to generate text.

""",

    """
User: What is training?
Assistant: Training is the process of adjusting a neural network's parameters using data so that its predictions become better.

""",

    """
User: What is a loss function?
Assistant: A loss function measures the difference between a model's prediction and the correct target.

""",

    """
User: What is an optimizer?
Assistant: An optimizer changes the parameters of a neural network to reduce the loss during training.

"""
]


# ==========================================
# 2. Add simple conversational examples
# ==========================================

conversations += [

    """
User: Hello
Assistant: Hello! I am your small language model.

""",

    """
User: Hi
Assistant: Hi! How can I help you?

""",

    """
User: Who are you?
Assistant: I am a small language model built using Python and a Transformer architecture.

""",

    """
User: What can you do?
Assistant: I can generate text based on patterns learned from my training data.

""",

    """
User: Are you an AI?
Assistant: Yes. I am a small artificial intelligence model trained to generate text.

""",

    """
User: I am learning Python.
Assistant: That is great! Python is a useful language for programming and artificial intelligence.

"""
]


# ==========================================
# 3. Shuffle conversations
# ==========================================

random.shuffle(conversations)


# ==========================================
# 4. Create larger dataset
# ==========================================

all_text = ""

for conversation in conversations:

    # Repeat each conversation several times
    all_text += conversation * 30


# ==========================================
# 5. Shuffle complete conversations
# ==========================================

sections = all_text.split("\n\n")

random.shuffle(sections)

all_text = "\n\n".join(sections)


# ==========================================
# 6. Train / validation split
# ==========================================

split = int(len(all_text) * 0.9)

train_text = all_text[:split]

val_text = all_text[split:]


# ==========================================
# 7. Save datasets
# ==========================================

with open(
    "chat_train.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(train_text)


with open(
    "chat_val.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(val_text)


# ==========================================
# 8. Show information
# ==========================================

print("Chat dataset created!")

print(
    "Total characters:",
    len(all_text)
)

print(
    "Training characters:",
    len(train_text)
)

print(
    "Validation characters:",
    len(val_text)
)

print(
    "Vocabulary size:",
    len(set(all_text))
)