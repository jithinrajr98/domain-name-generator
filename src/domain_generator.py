

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



class DomainNameGenerator:
    def __init__(self, base_model_name: str, adapter_path: str):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        print("Model loaded!")

    def calculate_confidence(self, domain: str, business_description: str) -> float:

        """Calculate confidence score for a domain (0-1 scale)"""
        domain_lower = domain.lower()
        business_words = set(business_description.lower().split())
        domain_words = set(re.findall(r'[a-z]+', domain_lower.split('.')[0]))
        
        # 1. Relevance score (word overlap)
        if business_words:
            overlap = len(business_words.intersection(domain_words))
            relevance_score = min(1.0, overlap / len(business_words) * 2)  # Scale to 0-1
        else:
            relevance_score = 0.5
        
        # 2. Length score (optimal 6-15 chars)
        domain_name = domain.split('.')[0]
        length = len(domain_name)
        if 6 <= length <= 15:
            length_score = 1.0
        elif 4 <= length <= 20:
            length_score = 0.7
        else:
            length_score = 0.3
        
        # 3. Complexity penalty
        complexity_penalty = 0
        if '-' in domain_name:
            complexity_penalty += 0.2
        if any(c.isdigit() for c in domain_name):
            complexity_penalty += 0.1
        if domain_name.lower() != domain_name:  # Mixed case
            complexity_penalty += 0.1
        
        # 4. Final confidence calculation
        confidence = (relevance_score * 0.5 + length_score * 0.3) * (1 - complexity_penalty)
        confidence = round(confidence, 3)
        return (max(0.1, min(1.0, confidence)))  # Clamp between 0.1 and 1.0
    
    def generate_domains(self, business_description: str, num_return_sequences: int = 3) -> List[str]:

        prompt = f"""Generate a list good domain name suggestions for a business described as: {business_description}\n\n and return only the domain names in json format
        ###Output: """      
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=150,
                num_return_sequences=num_return_sequences,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            output_part = generated_text.split("###Output:")[-1].strip()
            output_part = output_part.split(',')
            output_part = [item.strip() for item in output_part]
            
            domains = [(i,self.calculate_confidence(i, business_description)) for i in output_part]
            results.append(domains)

        sorted_list = [item for sublist in results for item in sublist]
        sorted_list = sorted(sorted_list, key=lambda x: x[1], reverse=True)


        return sorted_list[:3]
    
    

     
# # Initialize model
# BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ADAPTER_PATH = "/content/content/tinylamma_v2_qlora"
# generator = DomainNameGenerator(BASE_MODEL_NAME, ADAPTER_PATH)
