#!/usr/bin/env python3
"""
Pretrain 163M Parameter GPT Model on Turkish Dataset
Based on GPT-2 architecture with optimized hyperparameters for ~163M parameters
"""

import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from tqdm import tqdm
import math

# Import for mixed precision training
try:
    from torch.amp import autocast, GradScaler
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        MIXED_PRECISION_AVAILABLE = True
    except ImportError:
        MIXED_PRECISION_AVAILABLE = False
        print("⚠️  Mixed precision not available, using FP32")

class TurkishTextDataset(Dataset):
    """Dataset for Turkish text pretraining"""

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        print(f"📊 Total tokens in dataset: {len(token_ids):,}")

        # Create sliding windows
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def main():
    print("🇹🇷 Pretraining 163M Parameter GPT Model on Turkish Dataset (VRAM Optimized)")
    print("=" * 70)

    # Configuration for ~163M parameters - VRAM optimized
    config = {
        "vocab_size": 50257,     # GPT-2 vocabulary size
        "context_length": 512,   # Reduced context length for VRAM efficiency
        "emb_dim": 896,          # Embedding dimension (scaled up from 768)
        "n_heads": 14,           # Number of attention heads (scaled up from 12)
        "n_layers": 12,          # Number of transformer blocks (same as GPT-2 Small)
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
        "learning_rate": 3e-4,   # Slightly lower LR for stability
        "batch_size": 2,         # Small batch size for VRAM efficiency
        "num_epochs": 2,         # Fewer epochs for faster training
        "weight_decay": 0.1,
        "seed": 42,
        "max_length": 512,       # Reduced sequence length for VRAM efficiency
        "stride": 384,           # Adjusted stride for new sequence length
        "gradient_clip": 1.0,    # Gradient clipping value
        "mixed_precision": True  # Enable mixed precision for VRAM savings
    }

    torch.manual_seed(config["seed"])

    # Device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🖥️  Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🖥️  Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("🖥️  Using CPU")

    # Load Turkish dataset
    print("\n📖 Loading Turkish dataset...")
    dataset_file = "/Users/nurcunal/Bedai-Turkish-Base-LLM/fineweb_turkish_dataset/fineweb_turkish_sample copy.txt"

    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        turkish_text = f.read()

    print(f"📊 Dataset loaded: {len(turkish_text):,} characters")
    print(f"📊 Approximate tokens: {len(turkish_text.split()):,}")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    print("\n🔄 Creating training dataset...")
    train_dataset = TurkishTextDataset(turkish_text, tokenizer, config["max_length"], config["stride"])

    print(f"📊 Training sequences: {len(train_dataset):,}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    print(f"📊 Training batches per epoch: {len(train_loader):,}")

    # Import and initialize model
    print("\n🏗️ Initializing 163M GPT Model...")

    # Import GPT model architecture
    import sys
    sys.path.append('/Users/nurcunal/Bedai-Turkish-Base-LLM')
    from ch04_gpt_model import GPTModel

    model = GPTModel(config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(",")
    # Calculate approximate parameter count
    embedding_params = config["vocab_size"] * config["emb_dim"] + config["context_length"] * config["emb_dim"]
    transformer_params = config["n_layers"] * (
        # Attention parameters
        3 * config["emb_dim"] * config["emb_dim"] +  # Q, K, V projections
        config["emb_dim"] * config["emb_dim"] +      # Output projection
        # Feed-forward parameters
        4 * config["emb_dim"] * config["emb_dim"] +  # FF expansion
        config["emb_dim"] * config["emb_dim"] +      # FF contraction
        # Layer norms
        2 * 2 * config["emb_dim"]                    # 2 layer norms per block * 2 params each
    )
    output_params = config["emb_dim"] * config["vocab_size"]

    approx_params = embedding_params + transformer_params + output_params
    print(".1f")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )

    # Mixed precision scaler
    scaler = None
    autocast_device = None
    if config.get("mixed_precision", False) and MIXED_PRECISION_AVAILABLE:
        if torch.cuda.is_available():
            scaler = GradScaler()
            autocast_device = "cuda"
            print("🔥 Using mixed precision training (FP16) on CUDA")
        elif torch.backends.mps.is_available():
            # MPS doesn't support GradScaler in PyTorch 2.8, use FP32
            print("🔥 MPS detected - using full precision training (FP32)")
        else:
            print("🔥 Using full precision training (FP32)")
    else:
        print("🔥 Using full precision training (FP32)")

    # Training loop
    print("\n🚀 Starting Turkish GPT Pretraining...")
    print("-" * 50)

    start_time = time.time()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_batch, target_batch = batch
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Mixed precision training
            if scaler is not None and autocast_device is not None:
                with autocast(device_type=autocast_device):
                    logits = model(input_batch)
                    loss = torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1),
                        target_batch.flatten(),
                        ignore_index=-1
                    )

                # Scale loss and backward pass
                scaler.scale(loss).backward()

                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Full precision training
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    target_batch.flatten(),
                    ignore_index=-1
                )

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])

                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

            # Generate sample every 100 steps (more frequent due to smaller batches)
            if global_step % 100 == 0:
                sample = generate_turkish_sample(model, tokenizer, device, config, "Türkiye")
                print(f"\n🎨 Sample: {sample}\n")

        avg_loss = epoch_loss / num_batches
        print(".4f")
        # Save checkpoint
        checkpoint_path = f"turkish_gpt_163m_checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
            'global_step': global_step,
            'config': config
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = "turkish_gpt_163m_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'global_step': global_step,
                'config': config
            }, best_model_path)
            print(f"🏆 Best model saved: {best_model_path}")

    # Save final model
    final_model_path = "turkish_gpt_163m_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_loss': avg_loss,
        'total_epochs': config["num_epochs"],
        'dataset': 'FineWeb Turkish Sample Copy',
        'model_size': '163M'
    }, final_model_path)

    total_time = time.time() - start_time
    print(".2f")
    print("\n✅ Turkish GPT 163M pretraining completed!")
    print(f"💾 Final model saved as '{final_model_path}'")
    print("\n🎯 Your 163M parameter GPT model can now understand and generate Turkish text!")
    print("\n💡 Tips for next steps:")
    print("   - Fine-tune on specific tasks (classification, instruction following)")
    print("   - Evaluate model performance on Turkish benchmarks")
    print("   - Consider larger context lengths for better performance")


def generate_turkish_sample(model, tokenizer, device, config, prompt, max_new_tokens=30):
    """Generate sample Turkish text"""

    model.eval()

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = encoded_tensor[:, -config["context_length"]:]
            logits = model(context)
            logits = logits[:, -1, :]

            # Temperature sampling (optional)
            # logits = logits / 0.8  # temperature
            # probs = torch.softmax(logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)

            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            encoded_tensor = torch.cat((encoded_tensor, next_token), dim=1)

            # Stop at end of text token
            if next_token.item() == tokenizer.eot_token:
                break

    # Decode
    generated = encoded_tensor.squeeze(0).tolist()
    text = tokenizer.decode(generated)

    model.train()
    return text


if __name__ == "__main__":
    main()
