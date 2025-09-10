#!/usr/bin/env python3
"""
Chapter 4: Implementing a GPT Model from Scratch
Complete implementation of the GPT transformer architecture
"""

import torch
import torch.nn as nn
import tiktoken
import math
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version

print("🤖 Chapter 4: Implementing a GPT Model from Scratch")
print("=" * 60)

# Check versions
print(f"torch version: {version('torch')}")
print(f"tiktoken version: {version('tiktoken')}")
print()


class GPTDataset(Dataset):
    """Dataset for GPT training with sliding window approach"""

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Create sliding windows of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Linear layers for Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Output projection
        self.dropout = nn.Dropout(dropout)

        # Register causal mask as buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        # Linear transformations
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale and softmax
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context_vec = attn_weights @ values

        # Reshape back
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch_size, num_tokens, self.d_out)

        # Final output projection
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """Layer normalization"""

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation"""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feed-forward network with expansion factor"""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-layer norm"""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-layer norm + residual connection for attention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Pre-layer norm + residual connection for feed-forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """Complete GPT model architecture"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token and position embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final normalization and output head
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Simple text generation using greedy decoding"""

    for _ in range(max_new_tokens):
        # Crop context to supported length
        idx_cond = idx[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus on last time step
        logits = logits[:, -1, :]

        # Get most likely token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def create_dataloader(text, batch_size=4, max_length=256, stride=128,
                     shuffle=True, drop_last=True, num_workers=0):
    """Create data loader for GPT training"""

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def main():
    print("🏗️ Building GPT Model Architecture")
    print("-" * 40)

    # GPT-2 Small configuration (124M parameters)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # GPT-2 vocabulary size
        "context_length": 1024,  # Maximum context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of transformer blocks
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    print("Model Configuration:")
    for key, value in GPT_CONFIG_124M.items():
        print(f"  {key}: {value}")
    print()

    # Initialize model
    torch.manual_seed(42)
    model = GPTModel(GPT_CONFIG_124M)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(",")

    # Test forward pass
    print("🔍 Testing Forward Pass")
    print("-" * 25)

    batch_size = 2
    seq_len = 10
    test_input = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (batch_size, seq_len))

    print(f"Input shape: {test_input.shape}")

    with torch.no_grad():
        logits = model(test_input)

    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {GPT_CONFIG_124M['vocab_size']})")
    print()

    # Test text generation
    print("🎨 Testing Text Generation")
    print("-" * 30)

    model.eval()
    start_context = "The future of artificial intelligence is"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"Input text: '{start_context}'")
    print(f"Encoded: {encoded}")
    print(f"Input tensor shape: {encoded_tensor.shape}")

    # Generate text
    generated = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    generated_text = tokenizer.decode(generated.squeeze(0).tolist())
    print(f"Generated text: '{generated_text}'")
    print()

    # Test data loading
    print("📊 Testing Data Loading")
    print("-" * 25)

    # Load sample text
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()

    dataloader = create_dataloader(
        sample_text,
        batch_size=4,
        max_length=256,
        stride=128
    )

    print(f"Number of batches: {len(dataloader)}")

    # Get first batch
    first_batch = next(iter(dataloader))
    inputs, targets = first_batch

    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    # Test model on batch
    with torch.no_grad():
        batch_logits = model(inputs)

    print(f"Batch logits shape: {batch_logits.shape}")
    print()

    print("✅ GPT Model successfully implemented!")
    print("🎯 Model Architecture Summary:")
    print(f"   - {GPT_CONFIG_124M['n_layers']} transformer blocks")
    print(f"   - {GPT_CONFIG_124M['n_heads']} attention heads per block")
    print(f"   - {GPT_CONFIG_124M['emb_dim']} embedding dimensions")
    print(f"   - {GPT_CONFIG_124M['context_length']} maximum context length")
    print(f"   - {total_params:,} total parameters")
    print()
    print("🚀 Ready to move to Chapter 5: Pretraining!")


if __name__ == "__main__":
    main()
