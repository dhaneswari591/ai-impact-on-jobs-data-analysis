"""
AI Job Market Analysis
----------------------
This project analyzes the impact of AI on jobs using clustering, association rules mining, 
and network visualization. It uses real-world job sector data to explore AI-driven employment trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ---------------------------- Data Preprocessing ----------------------------

def load_data():
    """Loads and returns the job sector dataset as a Pandas DataFrame."""
    data = {
        "Job Sector": [
            "Medical", "Finance", "Retail", "Manufacturing", "Government",
            "Entertainment", "Agriculture", "Education", "Transport", "Telecom"
        ],
        "AI Created Jobs": [509, 453, 624, 789, 800, 754, 545, 454, 986, 994],
        "AI Displaced Jobs": [334, 440, 475, 598, np.nan, 568, 296, 274, 829, 854],
        "Unemployment Rate (%)": [5, 7, 8, 6, 4, 8, 6, 4, 5, 9],
        "Average Work Hours": [40, 35, 23, np.nan, 34, 75, 43, 57, 22, 21],
        "GDP Growth (%)": [4, 7, 5, 4, 7, 6, 5, 8, 7, 8],
        "Net Impact": [78, 77, 76, 45, 65, 34, 26, 76, 51, 87]
    }
    
    df = pd.DataFrame(data)
    
    # Handle missing values (Backward Fill Method)
    df.fillna(method='bfill', inplace=True)
    
    return df

# ---------------------------- K-Means Clustering ----------------------------

def apply_kmeans(df, n_clusters=3):
    """Applies K-Means clustering on job data and returns the updated DataFrame."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['AI Created Jobs', 'AI Displaced Jobs', 'Unemployment Rate (%)']])
    
    print("\n🔹 K-Means Clustering Results:")
    print(df[['Job Sector', 'Cluster']])
    
    return df

def plot_clusters(df):
    """Visualizes AI Job Impact Clustering using a scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Net Impact'], df['AI Created Jobs'], c=df['Cluster'], cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label="Cluster ID")
    plt.title("AI Job Impact Clustering")
    plt.xlabel("Net Impact")
    plt.ylabel("AI Created Jobs")
    plt.show()

# ---------------------------- Association Rule Mining ----------------------------

def association_rules_mining(df):
    """Performs Association Rules Mining on the dataset."""
    encoder = TransactionEncoder()
    df_encoded = encoder.fit(df.astype(str)).transform(df.astype(str))
    df_transformed = pd.DataFrame(df_encoded, columns=encoder.columns_)

    frequent_itemsets = apriori(df_transformed, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    print("\n🔹 Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ---------------------------- Network Graph Visualization ----------------------------

def plot_network_graph(df):
    """Creates a network graph to visualize AI's impact on job sectors."""
    edges = [(df['AI Created Jobs'][i], df['AI Displaced Jobs'][i]) for i in range(len(df))]
    G = nx.Graph()
    G.add_edges_from(edges)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
    plt.title("AI Job Impact Network Graph")
    plt.show()

# ---------------------------- Main Execution ----------------------------

if __name__ == "__main__":
    df = load_data()
    
    print("📊 Dataset Overview:")
    print(df.info())

    df = apply_kmeans(df)
    plot_clusters(df)

    association_rules_mining(df)
    plot_network_graph(df)
