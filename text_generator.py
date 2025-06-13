import torch
#import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
#import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import time

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token            # أو أضف [PAD] جديد

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tok.pad_token_id

def tokenize(batch):
    return tok(batch["text"],
               padding=True,
               truncation=True,
               return_attention_mask=True)

# ... بقية كود التحميل و Trainer

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        # تهيئة النموذج وأداة الترميز
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # إضافة رمز التبطين الخاص
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # تحديث حجم المصفوفة المضمنة لتتناسب مع الرمز الجديد
        self.model.resize_token_embeddings(len(self.tokenizer))
        # تعيين معرف رمز التبطين
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess_text(self, text):
        # معالجة النص مع التبطين الصحيح
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

    def generate_text(self, prompt, max_length=100, temperature=0.7):
        # توليد نص جديد باستخدام التشفير المحسن
        encoded_input = self.preprocess_text(prompt)
        
        output = self.model.generate(
            encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def fine_tune(self, training_texts, epochs=3, batch_size=4):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(training_texts), batch_size):
                batch = training_texts[i:i+batch_size]
                # معالجة الدفعة مع التبطين الصحيح
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(training_texts)}")

    def time_generation(self, prompt, max_length=100, temperature=0.7):
        # حساب الوقت المستغرق للتوليد
        start_time = time.time()
        generated_text = self.generate_text(prompt, max_length, temperature)
        end_time = time.time()
        generation_time = end_time - start_time
        
        return {
            'text': generated_text,
            'time_taken': generation_time
        }

    def test_model(self, prompt, max_length=100):
        # اختبار النموذج مع حساب الوقت
        result = self.time_generation(prompt, max_length)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result['text']}")
        print(f"Generation Time: {result['time_taken']:.2f} seconds\n")
        return result
