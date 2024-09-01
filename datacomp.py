import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/allenai-specter')
model = AutoModel.from_pretrained('sentence-transformers/allenai-specter')

# Load the dataset
df = pd.read_csv(r"C:\Users\tiahi\NSF REU\tokenizing\RAG\final_texts.csv")

# Define the columns to process
columns_to_process = ["title", "authors", "publication_date", "abstract", "introduction", "results", "discussion", "conclusion"]

# Define stopwords
stopwords = set([
    "the", "and", "to", "of", "a", "in", "that", "is", "for", "on", "with", "as", "by", "it", "this", "at", "from", "an",
    "are", "or", "be", "which", "was", "will", "has", "have", "can", "were", "not", "we", "but", "also", "their",
    "they", "our", "these", "been", "such", "may", "other", "more", "one", "there", "all", "would", "when", "if"
])

# Function to calculate total number of tokens
def total_tokens(df):
    total = 0
    for col in columns_to_process:
        texts = df[col].dropna().astype(str).tolist()
        for text in texts:
            tokens = tokenizer.tokenize(text)
            total += len(tokens)
    return total

# Function to find top 10 most reoccurring words (excluding numbers and stopwords)
def top_10_words(df):
    word_counter = Counter()
    for col in columns_to_process:
        texts = df[col].dropna().astype(str).tolist()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            filtered_words = [word for word in words if word not in stopwords and not word.isdigit()]
            word_counter.update(filtered_words)
    return word_counter.most_common(10)

# Get the top 10 words
top_words = top_10_words(df)

# Prepare data for plotting
words, counts = zip(*top_words)

# Plot the top 10 words
plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='red')
plt.title('Top 10 Most Reoccurring Alphanumeric Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
# Function to measure processing speed metrics
def measure_processing_speed():
    sample_text = "This is a sample text to measure the embedding generation time and response generation time."

    # Measure embedding generation time
    start_time = time.time()
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    embedding_generation_time = time.time() - start_time

    # Here we just simulate response generation time
    start_time = time.time()
    # Simulate a delay representing response generation (e.g., network latency, API response time)
    time.sleep(0.1)
    response_generation_time = time.time() - start_time

    return embedding_generation_time, response_generation_time

if __name__ == "__main__":
    # Calculate total number of tokens
    total_tokens_count = total_tokens(df)
    print(f"Total number of tokens: {total_tokens_count}")

    # Measure processing speed metrics
    embedding_gen_time, response_gen_time = measure_processing_speed()
    print(f"Embedding Generation Time: {embedding_gen_time:.4f} seconds")
    print(f"Response Generation Time: {response_gen_time:.4f} seconds")
