# ask_question.py
#
# Semantic similarity search against ChromaDB for library assistant example
#
# John Brunsfeld and Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# jbrunsfeld@uidaho.edu, sheneman@uidaho.edu
# 2024

import chromadb
from chromadb.config import Settings
import os
import json
import sys


def search(question):
	client = chromadb.PersistentClient(path="db/assistant.db")
	collection = client.get_collection(name="assistant")
	results = collection.query(
		query_texts=[question],
		n_results=1,
		# where={"metadata_field": "is_equal_to_this"}, # optional filter
		# where_document={"$contains":"search_string"}  # optional filter
	)
    
	metadatas = results["metadatas"]
	answer = metadatas[0][0]["answer"]
	return answer


client = chromadb.PersistentClient(path="db/assistant.db")
collection = client.get_collection(name="assistant")
results = collection.query(
	query_texts=["hello"],
	n_results=1,
	# where={"metadata_field": "is_equal_to_this"}, # optional filter
	# where_document={"$contains":"search_string"}  # optional filter
	)

print("Welcome to the University of Idaho Library!")

while True:
	question = input("--> How can I help you? ")
  
	if question.lower() == 'quit':
		break
    
	response = search(question)
  
	print(response)
	print("\n")

print("Goodbye!")
