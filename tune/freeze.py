# freeze.py
#
# An example of Parameter Efficient Fine Tuning
#
# Load GPT-2 from HuggingFace
# ----------------------------
#   1. Print the layers of the model
#   2. Freeze the first 3/4 transformer blocks of the model prior to fine-tuning
#   3. Print the layers of the model (again, now woth frozen transformer blocks)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# sheneman@uidaho.edu
# 2024
#

import torch
from transformers import GPT2LMHeadModel

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

params = model.parameters()

# Access the individual layers (modules) of the model
model_layers = list(model.children())

# let's print a condensed version of the model, starting with the first layer
first_layer = model_layers[0]
print("\n")
print("Condensed representation of GPT-2 Architecture:")
print("\n\n")
print(first_layer)
print("\n\n")

# Print the layers of the model
print("\n\n*******************************")
print("ORIGINAL MODEL:")
print("NUMBER OF TRANSFORMER BLOCKS: ", len(model.transformer.h))
print("NUMBER OF TRAINABLE PARAMETERS: ", sum(p.numel() for p in params if p.requires_grad))
print("*******************************\n")
for idx, (name, param) in enumerate(model.named_parameters()):
	print(f"Layer {idx}: {name}, Frozen: {not param.requires_grad}")



# Freeze the first 5/6 of the transformer blocks
num_layers = len(model.transformer.h)  # Total number of transformer blocks
num_layers_to_freeze = 5 * num_layers // 6  # Calculate 5/6 of the total

for layer in model.transformer.h[:num_layers_to_freeze]:
	for param in layer.parameters():
		param.requires_grad = False





print("\n\n*******************************")
print("FREEZING MODEL FOR FINE-TUNING")
print("*******************************\n")



#
# Here, let's explicitly freeze the token embedding layer
#
# Find the "wte" layer
wte_layer = None
for layer_name, layer in model.named_modules():
	if "wte" in layer_name:
		wte_layer = layer
		break

# Check if the "wte" layer was found
if wte_layer is not None:
	# Freeze the "wte" layer by setting requires_grad to False for its parameters
	for param in wte_layer.parameters():
		param.requires_grad = False





params = model.parameters()

# Print which layers are frozen and which are not
print("\n\n*******************************")
print("FROZEN MODEL:")
print("NUMBER OF TRANSFORMER BLOCKS: ", len(model.transformer.h))
print("NUMBER OF TRAINABLE PARAMETERS: ", sum(p.numel() for p in params if p.requires_grad))
print("*******************************\n")
for idx, (name, param) in enumerate(model.named_parameters()):
	print(f"Layer {idx}: {name}, Frozen: {not param.requires_grad}")



