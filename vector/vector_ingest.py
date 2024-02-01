# vector_ingest.py
#
# Use AI to compute semantic embeddings for prompts from library
# and injest them into a ChromaDB Vector DB
#
# John Brunsfeld and Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# jbrunsfeld@uidaho.edu, sheneman@uidaho.edu
# 2024
#


import chromadb
from chromadb.config import Settings
import os
import json
import sys

INPUT_FILE = "../data/prompt_response.json"

def ingest():
	client = chromadb.PersistentClient(path="db/assistant.db")

	#client.delete_collection(name="assistant")    # uncomment this to delete the assistant collection

	collection = client.get_or_create_collection(name="assistant")

	print(collection.count())

	with open(INPUT_FILE, "r") as file:
		data = json.load(file)

	print("Ingesting " + str(len(data)) + " prompts")


	for i, chunk in enumerate(data):
		questions = []
		metadatas = []
		ids	  = []

		prompt   = chunk["PROMPT"]
		response = chunk["RESPONSE"]

		questions.append(prompt)
		metadatas.append({'answer':response})
		ids.append(str(i))

		print("Ingesting prompt: " + str(i) )

		collection.add(
			documents = questions,
			metadatas = metadatas,
			ids = ids
		)

ingest()

