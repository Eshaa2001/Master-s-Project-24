import json
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from evaluate import load
import plotly.express as px
import numpy as np
from keybert import KeyBERT
from collections import defaultdict
import pandas as pd

# Initialize KeyBERT for keyword extraction
kw_model = KeyBERT()

# Load evaluation metrics
rouge = load("rouge")
bleu = load("bleu")

# Load dataset
with open("Data/fine_tuning_d1.json", "r") as f:
    data = json.load(f)

# Extract queries and answers
queries = [entry["query"] for entry in data]
answers = [entry["answer"] for entry in data]

# Step 1: Perform PCA on query embeddings
vectorizer = CountVectorizer(max_features=1000)  # Limit to 1000 features for simplicity
query_vectors = vectorizer.fit_transform(queries).toarray()

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(query_vectors)

# Step 2: K-Means Clustering and Keyword Extraction
num_clusters = 5  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced_vectors)

# Extract cluster-wise keywords
cluster_keywords = defaultdict(list)
for cluster_id in range(num_clusters):
    cluster_queries = [queries[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
    cluster_text = " ".join(cluster_queries)
    keywords = kw_model.extract_keywords(cluster_text, keyphrase_ngram_range=(1, 2), top_n=5)
    cluster_keywords[cluster_id] = [kw[0] for kw in keywords]

# Step 3: Calculate ROUGE and BLEU scores
# Ensure predictions and references have the same length
predictions = queries  # Replace with actual model predictions if available
references = [[answer] for answer in answers]  # Wrap each reference in a list

# Compute ROUGE scores
rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

# Compute BLEU score
bleu_score = bleu.compute(predictions=predictions, references=references)

# Display scores
print("\nROUGE Scores:")
for key, value in rouge_results.items():
    print(f"{key}: {value:.4f}")
print(f"\nBLEU Score: {bleu_score['bleu']:.4f}")

# Prepare data for Plotly
data_for_plot = {
    "PCA Component 1": reduced_vectors[:, 0],
    "PCA Component 2": reduced_vectors[:, 1],
    "Cluster": clusters,
    "Query": queries,
}

# Convert to DataFrame
df = pd.DataFrame(data_for_plot)

# Add cluster keywords for hover data
df["Cluster Keywords"] = df["Cluster"].map(lambda c: ", ".join(cluster_keywords[c]))

# Plot using Plotly
fig = px.scatter(
    df,
    x="PCA Component 1",
    y="PCA Component 2",
    color="Cluster",
    hover_data=["Query", "Cluster Keywords"],
    title="PCA-Reduced Embeddings and Clusters",
    labels={"Cluster": "Cluster ID"},
)

fig.update_layout(
    title_x=0.5,
    xaxis_title="PCA Component 1",
    yaxis_title="PCA Component 2",
    legend_title="Clusters",
    hoverlabel=dict(font_size=12, font_family="Arial"),
)

fig.show()
