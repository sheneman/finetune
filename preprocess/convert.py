# convert.py
#
# Process a 2-column version of the transcripts CSV
# and convert to an interim format  before calling the
# "makeprompt" scripts
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# sheneman@uidaho.edu
# 2024


import csv
import re

def process_csv(input_file, output_file):
	with open(input_file, mode='r', encoding='utf-8', newline='') as infile, \
		open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

	reader = csv.reader(infile)
	next(reader)  # Skip header row
	for row in reader:
		initial_question = row[0].strip()
		transcript = row[1].strip()

		# Remove control-M and UTF-8 characters
		transcript = re.sub(r'\^M|\r', '', transcript)
		transcript = transcript.encode('ascii', 'ignore').decode('ascii')

		# Prepend the initial question to the transcript
		combined_text = f"{initial_question}\n{transcript}\n\n***\n\n"

		outfile.write(combined_text)


process_csv('transcripts.csv', 'output.txt')



