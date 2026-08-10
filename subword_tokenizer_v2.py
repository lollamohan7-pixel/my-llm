import re


# ==========================================
# TOKENIZE TEXT
# ==========================================

def tokenize(text):

    # Keep words, numbers, punctuation and newlines
    parts = re.findall(
        r"[A-Za-z]+|[0-9]+|[.,!?]|\n",
        text
    )

    tokens = []

    for part in parts:

        # New line
        if part == "\n":
            tokens.append("\n")
            continue

        # Punctuation
        if part in [".", ",", "!", "?"]:
            tokens.append(part)
            continue

        # Convert word to lowercase
        word = part.lower()

        # Keep important known words as complete tokens
        known_words = {
            "user",
            "assistant",
            "python",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "neural",
            "network",
            "computer",
            "programming",
            "language",
            "model",
            "transformer",
            "attention",
            "algorithm",
            "data",
            "software",
            "hardware",
            "program",
            "memory",
            "processor",
            "electronics",
            "microcontroller",
            "internet",
            "things",
            "question",
            "answer",
            "hello",
            "what",
            "why",
            "how",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "and",
            "in",
            "on",
            "can",
            "do",
            "does",
            "used",
            "useful",
            "learn",
        }

        if word in known_words:

            tokens.append(
                "▁" + word
            )

        else:

            # Break unknown words into pieces
            # while preserving word start
            first = True

            i = 0

            while i < len(word):

                piece = word[i:i + 3]

                if first:

                    tokens.append(
                        "▁" + piece
                    )

                    first = False

                else:

                    tokens.append(piece)

                i += 3

    return tokens


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    tests = [
        "User: What is Python?",
        "Artificial intelligence is interesting.",
        "What is a neural network?",
        "Quantum computers are powerful."
    ]

    for sentence in tests:

        print()
        print("Original:")
        print(sentence)

        print()

        print("Tokens:")

        print(
            tokenize(sentence)
        )