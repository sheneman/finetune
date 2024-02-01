# finetune.py
#
# Fine-tune GPT2 on library transcripts - prompt/response
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# sheneman@uidaho.edu
# January, 2024
#

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import json


# some Hyperparameters
BATCH_SIZE     = 12
NUM_EPOCHS     = 3
LEARNING_RATE  = 5e-5
INPUT_FILE     = "../data/prompt_response.json"



# Load GPT-2 model and tokenizer
model_name = "gpt2"
model      = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer  = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Load and process data
with open(INPUT_FILE, 'r') as file:
	data = json.load(file)

train_data = []
for item in data:
	input_text = "Q: " + item['PROMPT'] + " A:"
	target_text = item['RESPONSE']
	train_data.append((input_text, target_text))

# Define a custom dataset
class QADataset(Dataset):
	def __init__(self, data, tokenizer):
		self.data = data
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		input_text, target_text = self.data[idx]
		encoding = self.tokenizer(input_text, target_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
		return {key: val.squeeze() for key, val in encoding.items()}

dataset = QADataset(train_data, tokenizer)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

number_of_batches = len(loader)

# Set up training
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()  # set model to "train" mode (PyTorch)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
	batch_idx = 0
	for batch in loader:

		optimizer.zero_grad()

		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['input_ids'].to(device)  # Labels are input ids

		outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
		loss = outputs.loss
		loss.backward()
		optimizer.step()

		# Report loss
		if(batch_idx % 100 == 0):
			print(f"Batch {batch_idx + 1}/{number_of_batches}, Loss: {loss.item()}")

		batch_idx += 1
        
	print(f"Epoch {epoch + 1} completed")

# Save the fine-tuned model
model.save_pretrained('fine_tuned_gpt2_qa_model')
tokenizer.save_pretrained('fine_tuned_gpt2_qa_model')

