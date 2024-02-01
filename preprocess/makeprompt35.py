# makeprompt35.py
#
# Parse our preprocessed library transcripts and send to OpenAI
# GPT 3.5 Turbo Instruct model via API endpoint
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


#
# Set some variables
#
API_KEY 	= "<API KEY>"
MODEL   	= "gpt-3.5-turbo-instruct"
INPUT_FILE 	= "../data/output.txt"
OUTPUT_FILE     = "finetune35.json"
DELIMITER       = "***"
TEMPERATURE     = 0.3

client = OpenAI(api_key=API_KEY)


#
# Sends prompt to GPT model at OpenAI
#
def query_gpt(prompt, model=MODEL, max_tokens=1000, temperature=TEMPERATURE):
	try:
		response = client.completions.create(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
		content = response.choices[0].text.strip()
		return content
	except Exception as e:
		print(f"An error occurred: {e}")
		return None



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

		prompt = "For the following chat dialog text, the first question is usually the Patron and the next line the Librarian.  Convert questions from Patron to prompts and convert answers from Librarian to responses following this example[ {\"PROMPT\": \"This is a question?\", \"RESPONSE\": \"This is the response.\"} ],   Remove all dates, remove all timestamps in the form ##:##:## AND remove individual names and REMOVE student IDs and convert HTML entity encoding to normal ASCII text. Format URLs in a simple readable style. Here is the TEXT: " + "\n\n" + combined_string + "\n"

		if(len(prompt)>3000):
			print("Prompt Length > 3000.   Skipping.")
			combined_string = ""
			continue

		response = query_gpt(prompt=prompt)
		if(response is None):
			print("GPT response is None.  Skipping.")
			combined_string = ""
			continue

		print(response)

		outfile.write(response + ",\n")
		outfile.flush()

		combined_string = ""

	else:
		combined_string += line
file.close()

outfile.write("[\n")
outfile.close()


