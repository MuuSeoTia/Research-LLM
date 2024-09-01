import csv
import json

def create_jsonl(csvFilePath, jsonlFilePath):
    # Set a large but safe limit for the maximum field size
    csv.field_size_limit(10**7)  # Adjust this number as needed

    # Open the CSV file and the JSONL file
    with open(csvFilePath, encoding='utf-8') as csvf, open(jsonlFilePath, 'w', encoding='utf-8') as jsonlf:
        csvReader = csv.DictReader(csvf)

        for rows in csvReader:
            # Write each row as a JSON object in the JSONL file
            jsonlf.write(json.dumps(rows) + '\n')

csvFilePath = r'C:\Users\tiahi\NSF REU\tokenizing\RAG\final_texts.csv'
jsonlFilePath = r'C:\Users\tiahi\NSF REU\tokenizing\RAG\final_texts.jsonl'

# Call the create_jsonl function
create_jsonl(csvFilePath, jsonlFilePath)
