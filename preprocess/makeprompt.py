#
# makeprompt.py
#
# Parse our preprocessed library transcripts and send to GPT-4
# via OpenAI API endpoint
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# sheneman@uidaho.edu
# 2024
#

from openai import OpenAI
import time

API_KEY 	= "<API KEY>"
MODEL   	= "gpt-4"
INPUT_FILE 	= "output.txt"
OUTPUT_FILE     = "finetune.json"
DELIMITER       = "***"
TEMPERATURE     = 0.5

client = OpenAI(api_key=API_KEY)



#
# Sends prompt to GPT model at OpenAI
#
def query_gpt(messages, model=MODEL, max_tokens=1000, temperature=TEMPERATURE):
	try:
		response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
		choices = response.choices
		chat_completion = choices[0]
		content = chat_completion.message.content # Correct (this works with the Chat Completions API)
		return content
	except Exception as e:
		print(f"An error occurred: {e}")
		return None

outfile = open(OUTPUT_FILE, "w")
outfile.write("[\n")



#
# MAIN LOOP
#
# Using the characters "***" as a delimeter between different transcript exchanges
# Have the LLM parse, clean, and re-format our data into JSON
#
file = open(INPUT_FILE, "r")
combined_string = ""
for line in file:
	if line.strip() == DELIMITER:
		#print("STRING: ", combined_string, "\n\n")

		messages = [
			{"role": "system", "content":"You are a helpful assistant that only outputs JSON"},
			{"role": "user", "content":"Respond in JSON format: reformat questions and answers into prompt-response pairs in JSON format suitable for fine-tuning a conversational LLM at OpenAI. Each pair should follow this example: [ {\"prompt\": \"This is a question?\", \"response\": \"This is the response.\"} ],  Let's break this down and solve it step by step.  For each sentence in the following text, only sentences ending in a question mark can be prompt.   ALWAYS Remove all dates and timestamps AND remove individual names and REMOVE student IDs and convert HTML entity encoding to normal ASCII text. Format URLs in a simple readable style. Only output JSON-formatted text.  TEXT: " + "\n\n" + combined_string + "\n"}
		]

		response = query_gpt(messages=messages)
		#print("\n", combined_string, "\n")
		print(response)

		outfile.write(response + ",\n")
		outfile.flush()

		combined_string = ""

	else:
		combined_string += line
file.close()

outfile.write("[\n")
outfile.close()



