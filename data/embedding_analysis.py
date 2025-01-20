import numpy as np
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os

class PhilosophicalSpaceAnalyzer:
    def __init__(self, embeddings_path: str, metadata_path: str):
        # Load embeddings and metadata
        self.embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Create DataFrame for easier analysis
        self.df = pd.DataFrame({
            'framework': self.metadata['frameworks_per_sample'],
            'author': self.metadata['authors'],
            'work': self.metadata['works']
        })
        
    def get_framework_embeddings(self, framework: str) -> np.ndarray:
        """Get embeddings for a specific framework."""
        framework_indices = self.df['framework'] == framework
        return self.embeddings[framework_indices]
    
    def get_author_embeddings(self, author: str) -> np.ndarray:
        """Get embeddings for a specific author."""
        author_indices = self.df['author'] == author
        return self.embeddings[author_indices]
    
    def compute_mean_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """Compute mean cosine similarity between two sets of embeddings."""
        similarities = cosine_similarity(embeddings1, embeddings2)
        # Scale similarities to [0, 1] range
        similarities = (similarities + 1) / 2
        
        # Get mean similarity excluding self-similarities if comparing same set
        if embeddings1 is embeddings2:
            np.fill_diagonal(similarities, np.nan)
            return np.nanmean(similarities)
        return np.mean(similarities)
    
    def plot_framework_clusters(self):
        """Plot frameworks in 2D space."""
        plt.figure(figsize=(12, 8))
        frameworks = self.df['framework'].unique()
        
        for framework in frameworks:
            mask = self.df['framework'] == framework
            plt.scatter(self.embeddings[mask, 0], self.embeddings[mask, 1], 
                       label=framework, alpha=0.6)
        
        plt.title('Framework Clusters')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def analyze_framework_similarities(self):
        """Analyze similarities between different frameworks."""
        frameworks = self.df['framework'].unique()
        n_frameworks = len(frameworks)
        similarity_matrix = np.zeros((n_frameworks, n_frameworks))
        
        for i, f1 in enumerate(frameworks):
            for j, f2 in enumerate(frameworks):
                emb1 = self.get_framework_embeddings(f1)
                emb2 = self.get_framework_embeddings(f2)
                similarity_matrix[i, j] = self.compute_mean_similarity(emb1, emb2)
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, 
                   xticklabels=frameworks, 
                   yticklabels=frameworks,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu')
        plt.title('Framework Similarities')
        plt.tight_layout()
        plt.show()
        
        return similarity_matrix, frameworks
    
    def analyze_authors_in_framework(self, framework: str):
        """Analyze similarities between authors within a framework."""
        framework_authors = self.df[self.df['framework'] == framework]['author'].unique()
        n_authors = len(framework_authors)
        similarity_matrix = np.zeros((n_authors, n_authors))
        
        for i, a1 in enumerate(framework_authors):
            for j, a2 in enumerate(framework_authors):
                emb1 = self.get_author_embeddings(a1)
                emb2 = self.get_author_embeddings(a2)
                similarity_matrix[i, j] = self.compute_mean_similarity(emb1, emb2)
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, 
                   xticklabels=framework_authors, 
                   yticklabels=framework_authors,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu')
        plt.title(f'Author Similarities within {framework}')
        plt.tight_layout()
        plt.show()
        
        return similarity_matrix, framework_authors
    
    def find_most_similar_excerpts(self, query_idx: int, n: int = 5):
        """Find n most similar excerpts to a given excerpt."""
        query_embedding = self.embeddings[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        similarities = (similarities + 1) / 2  # Scale to [0, 1]
        
        # Get top n most similar indices (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        print(f"\nQuery excerpt from {self.df.iloc[query_idx]['author']} ({self.df.iloc[query_idx]['framework']}):")
        print(f"Work: {self.df.iloc[query_idx]['work']}")
        
        print(f"\nMost similar excerpts:")
        for idx in similar_indices:
            similarity = similarities[idx]
            print(f"\nSimilarity: {similarity:.3f}")
            print(f"Author: {self.df.iloc[idx]['author']}")
            print(f"Framework: {self.df.iloc[idx]['framework']}")
            print(f"Work: {self.df.iloc[idx]['work']}")

def main():
    analyzer = PhilosophicalSpaceAnalyzer(
        embeddings_path="embedded_space/philosophical_space.npy",
        metadata_path="embedded_space/metadata.json"
    )
    
    # Plot framework clusters
    print("\nPlotting framework clusters...")
    analyzer.plot_framework_clusters()
    
    # Analyze framework similarities
    print("\nAnalyzing framework similarities...")
    similarity_matrix, frameworks = analyzer.analyze_framework_similarities()
    
    # Analyze authors within Utilitarianism
    print("\nAnalyzing authors within Utilitarianism...")
    analyzer.analyze_authors_in_framework("Utilitarianism")
    
    # Find similar excerpts for a random utilitarian excerpt
    utilitarian_indices = analyzer.df[analyzer.df['framework'] == 'Utilitarianism'].index
    if len(utilitarian_indices) > 0:
        random_idx = np.random.choice(utilitarian_indices)
        print("\nFinding similar excerpts to a random utilitarian excerpt...")
        analyzer.find_most_similar_excerpts(random_idx)

if __name__ == "__main__":
    main() 