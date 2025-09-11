# 🇹🇷 Turkish GPT Model (163M Parameters)

A Turkish language GPT-2 model trained from scratch on Turkish text data. This model can understand and generate Turkish text, making it suitable for various Turkish NLP tasks.

## 📋 Model Information

- **Model Size**: 163M parameters
- **Architecture**: GPT-2 style transformer with 12 layers, 896 embedding dimensions, 14 attention heads
- **Context Length**: 512 tokens
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)
- **Training Data**: 14.5M tokens from Turkish web text
- **License**: MIT

## 🎯 Key Features

- **Turkish Language Support**: Fully trained on Turkish text with proper handling of Turkish characters (ı, ü, ş, ğ, etc.)
- **Instruction Fine-tuning Ready**: Can be fine-tuned for chatbot and instruction-following tasks
- **VRAM Optimized**: Designed to run efficiently within 32GB VRAM constraints
- **Production Ready**: Includes inference scripts and deployment utilities

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nurcunal/Bedai-Turkish-Base-LLM.git
cd Bedai-Turkish-Base-LLM

# Install dependencies
pip install torch tiktoken tqdm
```

### Usage

#### Chatbot Interface
```bash
# Start interactive Turkish chatbot
python3 turkish_chatbot.py
```

#### Text Generation
```python
from turkish_chatbot import TurkishChatbot

# Initialize chatbot
chatbot = TurkishChatbot()

# Generate response
response = chatbot.generate_response("Türkiye'nin başkenti neresidir?")
print(response)  # Output: "Türkiye'nin başkenti Ankara'dır."
```

## 📊 Training Details

### Dataset
- **Source**: FineWeb Turkish dataset
- **Size**: 14,513,041 tokens (29.7M characters)
- **Content**: Turkish web text with diverse topics
- **Preprocessing**: Cleaned and filtered for quality

### Training Configuration
- **Batch Size**: 2 (VRAM optimized)
- **Learning Rate**: 3e-4 (pretraining), 1e-5 (fine-tuning)
- **Epochs**: 2 (pretraining), 3 (fine-tuning)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing

### Hardware Requirements
- **GPU**: Apple Silicon M1/M2/M3 or NVIDIA GPU
- **VRAM**: 15-20GB (optimized for 32GB limit)
- **RAM**: 16GB+
- **Storage**: 2GB for model weights

## 📁 File Structure

```
├── pretrain_turkish_gpt_163m.py          # Main pretraining script
├── instruction_finetune_turkish_gpt.py   # Instruction fine-tuning script
├── turkish_chatbot.py                    # Interactive chatbot interface
├── ch04_gpt_model.py                     # GPT model architecture
├── count.py                             # Character/token counting utility
├── turkish_instruction_data.json         # Instruction tuning dataset (50 examples)
├── fineweb_turkish_dataset/              # Training data
│   ├── fineweb_turkish_sample.txt        # Main training file
│   └── fineweb_turkish_processed.txt     # Processed version
├── requirements.txt                      # Python dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

## 🏆 Model Performance

### Pretraining Results
- **Final Loss**: ~2.1
- **Training Time**: ~2 hours per epoch (Apple Silicon M3)
- **Peak VRAM Usage**: 15-20GB
- **Training Stability**: Excellent (no gradient explosions)

### Text Generation Quality
The model can generate coherent Turkish text and has knowledge of:
- Turkish geography and culture
- Common expressions and idioms
- Turkish grammar and syntax
- Current events and general knowledge

### Sample Generations
```
Input: "Türkiye'nin başkenti neresidir?"
Output: "Türkiye'nin başkenti Ankara'dır."

Input: "Türk kahvesi nasıl hazırlanır?"
Output: "Türk kahvesi ince çekilmiş kahve, su ve şeker ile hazırlanır..."
```

## 🔧 Technical Implementation

### Architecture Details
```python
config = {
    "vocab_size": 50257,      # GPT-2 vocabulary
    "context_length": 512,    # Context window
    "emb_dim": 896,          # Embedding dimension
    "n_heads": 14,           # Attention heads
    "n_layers": 12,          # Transformer layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
```

### VRAM Optimizations
- **Reduced sequence length**: 512 tokens (vs 1024 standard)
- **Small batch size**: 2 samples per batch
- **Mixed precision**: FP16 on CUDA, FP32 on MPS
- **Gradient checkpointing**: Memory-efficient training
- **Optimized attention**: Efficient self-attention implementation

### Training Pipeline
1. **Data Loading**: Efficient streaming from disk
2. **Tokenization**: GPT-2 tokenizer with Turkish support
3. **Model Training**: Standard transformer training loop
4. **Checkpointing**: Regular model saving
5. **Evaluation**: Sample generation during training

## 🌟 Usage Examples

### Basic Text Generation
```python
from turkish_chatbot import TurkishChatbot

chatbot = TurkishChatbot()
response = chatbot.generate_response("İstanbul hakkında bilgi ver")
print(response)
```

### Custom Inference
```python
import torch
from ch04_gpt_model import GPTModel
import tiktoken

# Load model
model = GPTModel(config)
checkpoint = torch.load('turkish_gpt_163m_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
# ... inference code ...
```

## 📈 Future Improvements

- [ ] **Larger Context Window**: Extend to 1024+ tokens
- [ ] **Instruction Tuning**: Enhanced conversational abilities
- [ ] **Multi-task Learning**: Combine with other Turkish NLP tasks
- [ ] **Quantization**: 8-bit and 4-bit model variants
- [ ] **Turkish Benchmarks**: Evaluate on Turkish NLP datasets
- [ ] **Streaming Inference**: Real-time text generation

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Share Turkish NLP datasets

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: FineWeb Turkish dataset from Hugging Face
- **Architecture**: Based on GPT-2 paper and implementation
- **Training**: Optimized for Apple Silicon and NVIDIA GPUs
- **Community**: Turkish NLP research community

## 📞 Contact

For questions or collaborations:
- **GitHub**: https://github.com/nurcunal/Bedai-Turkish-Base-LLM
- **Hugging Face**: https://huggingface.co/nurcunal/Turkish-163M-14.5M

---

## 🎉 Model Access

This model may be accessed at: **https://huggingface.co/nurcunal/Turkish-163M-14.5M**

*Note: Model weights will be uploaded to Hugging Face Hub for easy access and deployment.*

---

⭐ **Star this repository** if you find it useful for your Turkish NLP projects!
