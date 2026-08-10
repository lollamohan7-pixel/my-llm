import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# CAUSAL SELF ATTENTION
# ==========================================

class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_size, block_size):
        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.key = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.value = nn.Linear(
            embedding_size,
            embedding_size
        )

        # Prevent looking into the future
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (
            Q.shape[-1] ** 0.5
        )

        length = x.shape[1]

        mask = self.mask[
            :length,
            :length
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        return attention @ V


# ==========================================
# TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        block_size
    ):
        super().__init__()

        self.attention = CausalSelfAttention(
            embedding_size,
            block_size
        )

        self.norm1 = nn.LayerNorm(
            embedding_size
        )

        self.norm2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = nn.Sequential(

            nn.Linear(
                embedding_size,
                embedding_size * 4
            ),

            nn.GELU(),

            nn.Linear(
                embedding_size * 4,
                embedding_size
            )
        )

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(
            self.norm1(x)
        )

        # Feed forward + residual
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


# ==========================================
# GPT v5
# ==========================================

class GPTv5(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size=192,
        block_size=128,
        num_blocks=4
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Position embedding
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_size
        )

        # Multiple Transformer blocks
        self.blocks = nn.Sequential(

            *[
                TransformerBlock(
                    embedding_size,
                    block_size
                )

                for _ in range(num_blocks)
            ]
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(
            embedding_size
        )

        # Output vocabulary prediction
        self.output = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, x):

        batch_size, sequence_length = x.shape

        # Token embeddings
        token_embeddings = (
            self.token_embedding(x)
        )

        # Positions
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        # Combine token + position
        x = (
            token_embeddings
            + position_embeddings
        )

        # Transformer
        x = self.blocks(x)

        # Normalize
        x = self.final_norm(x)

        # Predict next token
        logits = self.output(x)

        return logits


# ==========================================
# TEST MODEL
# ==========================================

if __name__ == "__main__":

    model = GPTv5(
        vocab_size=100,
        embedding_size=192,
        block_size=128,
        num_blocks=4
    )

    test_input = torch.tensor([
        [10, 20, 30, 40, 50]
    ])

    output = model(test_input)

    print("Input shape:")
    print(test_input.shape)

    print()

    print("Output shape:")
    print(output.shape)

    print()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Number of parameters:")
    print(parameters)