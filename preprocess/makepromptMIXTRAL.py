#
# makepromptMIXTRAL.py
#
# Parse our preprocessed library transcripts and send to Mixtral 8X7B model 
# via an OpenAI-compatible API endpoint
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# sheneman@uidaho.edu
# 2024
#

import requests
import json


#
# Set some variables
#
URL		= "<API ENDPOINT RUNNING LLAMA.CPP SERVER>"
API_KEY 	= None
MODEL   	= "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
INPUT_FILE 	= "output.txt"
OUTPUT_FILE     = "finetuneMIXTRAL.json"
DELIMITER       = "***"
TEMPERATURE     = 0.5


#
# Performs a JSON post to any OpenAI compatible API endpoint
#
def post_to_openai_compatible_endpoint(url, data):
	headers = {
		"Content-Type": "application/json",
		"Authorization": "Bearer no-key"
	}

	response = requests.post(url, headers=headers, data=json.dumps(data))
	if response.status_code == 200:
		return response.json()
	else:
		return response.text


#
# MAIN LOOP
# 
# Using the characters "***" as a delimeter between different transcript exchanges
# Have the LLM parse, clean, and re-format our data into JSON
#
outfile = open(OUTPUT_FILE, "w")
outfile.write("[\n")
file = open(INPUT_FILE, "r")
combined_string = ""
for line in file:
	if line.strip() == DELIMITER:
		#print("STRING: ", combined_string, "\n\n")

		data = {
			"model": MODEL,
			"temperature": TEMPERATURE,
			"messages": [
			{"role": "system", "content":"You are a helpful assistant that only outputs JSON"},
			{"role": "user", "content":"Respond in JSON format: reformat questions and answers into prompt-response pairs in JSON format suitable for fine-tuning a conversational LLM at OpenAI. Each pair should follow this example: [ {\"prompt\": \"This is a question?\", \"response\": \"This is the response.\"} ],  Let's break this down and solve it step by step.  For each sentence in the following text, only sentences ending in a question mark can be prompt.   ALWAYS Remove all dates and timestamps AND remove individual names and REMOVE student IDs and convert HTML entity encoding to normal ASCII text. Format URLs in a simple readable style. Only output JSON-formatted text.  TEXT: " + "\n\n" + combined_string + "\n"}
			]
		}


		response = post_to_openai_compatible_endpoint(URL, data).get('choices')[0].get('message').get('content') 
		print(response)
		
		outfile.write(response + ",\n")
		outfile.flush()

		combined_string = ""

	else:
		combined_string += line
file.close()

outfile.write("[\n")
outfile.close()



