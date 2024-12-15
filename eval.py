import json
import numpy as np
from sklearn.cluster import KMeans
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the fine-tuned model and tokenizer
fine_tuned_model_path = "C:/Users/eshaa/OneDrive/Documents/MFP/fine_tuned_model6/fine_tuned_model6/final_checkpoint"
tokenizer = BartTokenizer.from_pretrained(fine_tuned_model_path)
model = BartForConditionalGeneration.from_pretrained(fine_tuned_model_path)

# Load sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize KeyBERT for keyword extraction
kw_model = KeyBERT()

# Load the dataset
with open("C:/Users/eshaa/OneDrive/Documents/MFP/Data/fine_tuning_d1.json", "r") as f:
    dataset = json.load(f)

# Prepare data
queries = [f"Query: {item['query']} Context: {item['context']}" for item in dataset]
references = [item["answer"] for item in dataset]

# Generate embeddings for queries
query_embeddings = embedding_model.encode(queries, convert_to_tensor=False)

# KMeans Clustering to select representative samples
num_clusters = 10  # Adjust clusters based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(query_embeddings)

# Extract cluster-wise keywords
def extract_keywords_by_cluster(clusters, queries, num_keywords=5):
    cluster_keywords = defaultdict(list)
    for cluster_id in range(num_clusters):
        cluster_queries = [queries[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        cluster_text = " ".join(cluster_queries)
        keywords = kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 2), top_n=num_keywords)
        cluster_keywords[cluster_id] = [kw[0] for kw in keywords]
    return cluster_keywords

cluster_keywords = extract_keywords_by_cluster(clusters, queries)

# Select one representative sample per cluster
selected_indices = []
for cluster_id in range(kmeans.n_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    representative_index = cluster_indices[0]  # Take the first item in each cluster
    selected_indices.append(representative_index)

# Subset of queries and references
selected_queries = [queries[i] for i in selected_indices]
selected_references = [references[i] for i in selected_indices]

# Generate predictions for the selected subset
def generate_predictions(queries):
    predictions = []
    for query in queries:
        inputs = tokenizer(query, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(inputs["input_ids"], max_length=200, num_beams=4)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
    return predictions

selected_predictions = generate_predictions(selected_queries)

# Evaluate with ROUGE
rouge = evaluate.load("rouge")
rouge_scores = rouge.compute(predictions=selected_predictions, references=selected_references)

# Evaluate with BLEU
bleu = evaluate.load("bleu")
bleu_references = [[ref] for ref in selected_references]  # BLEU expects references as lists of lists
bleu_score = bleu.compute(predictions=selected_predictions, references=bleu_references)

# Print scores
print("ROUGE Scores:")
for key, value in rouge_scores.items():
    print(f"{key}: {value:.4f}")

print(f"\nBLEU Score: {bleu_score['bleu']:.4f}")

# Visualization of metrics
def plot_metrics(rouge_scores, bleu_score):
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
    scores = [
        rouge_scores["rouge1"],
        rouge_scores["rouge2"],
        rouge_scores["rougeL"],
        bleu_score["bleu"],
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, scores, color=['blue', 'green', 'red', 'purple'])
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics on Selected Subset")
    plt.ylabel("Score")
    plt.xlabel("Metrics")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=12)
    plt.show()

plot_metrics(rouge_scores, bleu_score)
