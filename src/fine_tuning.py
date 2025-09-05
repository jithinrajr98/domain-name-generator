import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from typing import List, Dict, Any

class DomainNameTrainer:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_tokenizer(self):
        """Initialize and configure the tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def tokenize_function(self, examples):
        """Tokenization function for the dataset"""
        full_texts = []
        
        for i in range(len(examples['business_description'])):
            prompt = f"### Instruction:\nGenerate a list of good domain name suggestions for a business described as: {examples['business_description'][i]}\n\n### Output:\n"
            target = ", ".join(examples['domain_suggestions'][i])
            full_text = prompt + target + self.tokenizer.eos_token
            full_texts.append(full_text)
        
        tokenized_output = self.tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        
        return tokenized_output
    
    def setup_model(self):
        """Initialize and configure the model with LoRA"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Freeze all original parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Wrap with PEFT
        self.model = get_peft_model(model, lora_config)
        print("Trainable parameters of the model:")
        self.model.print_trainable_parameters()
        
        return self.model
    
    def setup_trainer(self, train_dataset, output_dir: str = "./results"):
        """Setup the trainer with training arguments"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            logging_steps=10,
            save_steps=50,
            learning_rate=2e-3,
            fp16=False,
            push_to_hub=False,
            remove_unused_columns=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        return self.trainer
    
    def load_and_preprocess_data(self, data_path: str = "./domain_dataset.jsonl"):
        """Load and preprocess the dataset"""
        dataset = load_dataset("json", data_files=data_path)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["business_description", "domain_suggestions"])
        return tokenized_dataset
    
    def train(self, data_path: str = "./domain_dataset.jsonl", output_dir: str = "./results"):
        """Complete training pipeline"""
        print("Setting up tokenizer...")
        self.setup_tokenizer()
        
        print("Loading and preprocessing data...")
        tokenized_dataset = self.load_and_preprocess_data(data_path)
        
        print("Setting up model...")
        self.setup_model()
        
        print("Setting up trainer...")
        self.setup_trainer(tokenized_dataset["train"], output_dir)
        
        print("Starting training...")
        training_results = self.trainer.train()
        
        print("Saving model...")
        self.trainer.save_model(output_dir)
        
        print("Training completed!")
        return training_results
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.trainer:
            self.trainer.save_model(path)
        elif self.model:
            self.model.save_pretrained(path)
        print(f"Model saved to {path}")

# Usage example
if __name__ == "__main__":
    # Initialize the trainer
    trainer = DomainNameTrainer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Run the complete training pipeline
    results = trainer.train(
        data_path="./data/domain_dataset_v1.jsonl",
        output_dir="tinylamma_v1"
    )
    
 