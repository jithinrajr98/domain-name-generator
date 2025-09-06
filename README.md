# Domain Name Generation with QLoRA Fine-tuning

A comprehensive domain name generation system that uses QLoRA fine-tuning on TinyLlama-1.1B to create relevant, brandable domain suggestions for business descriptions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This project implements a domain name generation system using QLoRA (Quantized Low-Rank Adaptation) fine-tuning on the TinyLlama-1.1B-Chat model. The system generates creative, relevant domain names based on business descriptions while maintaining safety filtering and comprehensive evaluation metrics.

## Features

- **QLoRA Fine-tuning**: Memory-efficient fine-tuning using 4-bit quantization
- **Multi-version Dataset**: Three dataset versions for systematic performance evaluation
- **Safety Filtering**: Content moderation to block inappropriate business types
- **Comprehensive Evaluation**: LLM judge scoring and custom confidence metrics
- **Edge Case Handling**: Robust testing across various business description types
- **Production Ready**: Modular architecture with clear separation of concerns

## Project Structure

```
├── data_generator.py          # Synthetic dataset generation
├── fine_tuning.py            # QLoRA training pipeline
├── domain_generator.py       # Trained model inference
├── testing.py               # Comprehensive test framework
├── evaluation/
│   ├── domain_dataset_v1.jsonl  # 100 synthetic samples
│   ├── domain_dataset_v2.jsonl  # 500 synthetic samples
│   └── domain_dataset_v3.jsonl  # 290 LLM-generated samples
├── models/
│   ├── tinylamma_v1/        # Model 1 checkpoints
│   ├── tinylamma_v2/        # Model 2 checkpoints
│   └── tinylamma_v3/        # Model 3 checkpoints
└── results/                 # Evaluation results and reports
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT
- BitsAndBytes
- Pandas, NumPy

### Setup
```bash
git clone <repository-url>
cd domain-name-generator
pip install -r requirements.txt
```

### Dependencies
```bash
pip install torch transformers peft bitsandbytes datasets pandas numpy
```

## Usage

### 1. Generate Training Dataset
```python
from data_generator import DomainDatasetGenerator

generator = DomainDatasetGenerator()
dataset = generator.generate_dataset(size=1000)
generator.save_dataset(dataset, "domain_dataset.jsonl")
```

### 2. Fine-tune Model
```python
from fine_tuning import DomainNameTrainer

trainer = DomainNameTrainer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
results = trainer.train(
    data_path="./data/domain_dataset_v1.jsonl",
    output_dir="./models/tinylamma_v1"
)
```

### 3. Generate Domain Names
```python
from domain_generator import DomainNameGenerator

generator = DomainNameGenerator(
    base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path="./models/tinylamma_v3"
)

domains = generator.generate_domains("organic coffee shop in downtown area")
print(domains)  # [(domain, confidence_score), ...]
```

### 4. Run Evaluation
```python
from testing import TestFramework

test_framework = TestFramework(generator, safety_filter, judge)
results_df = test_framework.run_test()
test_framework.print_summary()
```

## Model Performance

### Overall Performance Comparison

| Metric | Model 1 | Model 2 | Model 3 | Improvement |
|--------|---------|---------|---------|-------------|
| Average LLM Score | 5.00/10 | 5.16/10 | 6.21/10 | +24.2% |
| Model Confidence | 0.303 | 0.474 | 0.437 | +44.2% |
| Best Individual Score | 6.7/10 | 6.6/10 | 7.0/10 | +4.5% |
| Output Quality | Poor | Good | Excellent | Significant |

### Edge Case Performance (Model 1 → Model 3)
- **Short Prompts**: 2.6/10 → 3.87/10 (+49%)
- **Ambiguous Descriptions**: 2.9/10 → 3.87/10 (+33%)
- **Technical Domains**: 5.27/10 → 6.87/10 (+30%)

## Dataset

### Three Dataset Versions
1. **v1**: 100 synthetic samples - Quick experimentation
2. **v2**: 500 synthetic samples - More training data
3. **v3**: 290 LLM-generated samples - Higher quality/diversity

### Generation Strategy
- **Business Types**: 20 categories (restaurant, tech startup, consulting firm, etc.)
- **Adjectives**: 21 descriptors (modern, premium, sustainable, etc.)
- **Locations**: 22 global cities
- **Domain Strategies**: Keyword combination, acronyms, hyphenation, common affixes

## Methodology

### QLoRA Configuration
```python
# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Safety Filter
- **Keywords**: 21 inappropriate terms (adult, gambling, violence, etc.)
- **Processing**: Case-insensitive matching with immediate blocking
- **Feedback**: Detailed rejection reasons for transparency

### Evaluation Metrics
- **Model Confidence**: Custom heuristic (relevance, length, complexity)
- **LLM Judge**: llama-3.3-70b evaluation on 5 criteria
- **Test Categories**: Normal, short, long, technical, unusual, multilingual, ambiguous

## Results

### Key Findings
- **Progressive Improvement**: Each model iteration showed measurable enhancements
- **Formatting Resolution**: Models 2 and 3 resolved critical output formatting issues
- **Category Performance**: Best with detailed descriptions, struggled with ambiguous prompts
- **Safety Consistency**: All models maintained identical safety filtering

### Production Recommendations
- **Deploy Model 3** for production use
- **Future Improvements**: Larger base model (7B+), enhanced LoRA configuration, real-world training data
- **QLoRA Enhancements**: Expand target modules, increase rank to r=32-64

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- TinyLlama team for the base model
- Hugging Face for transformers and PEFT libraries
- QLoRA authors for the quantization technique