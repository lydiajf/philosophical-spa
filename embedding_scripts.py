import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import numpy as np
import os
import json
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

class PhilosophicalEmbedder:
    def __init__(
        self, 
        model_name: str = "allenai/longformer-base-4096",
        pooling_strategy: str = "mean",
        n_components: int = 50  # Number of PCA components
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name)  # No need for output_hidden_states
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.pooling_strategy = pooling_strategy
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
    
    def _pool_tokens(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings into a single vector."""
        if self.pooling_strategy == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_strategy == "cls":
            return token_embeddings[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def embed_text(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for input text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize texts
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096,  # Longformer can handle up to 4096 tokens
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                last_hidden_state = outputs.last_hidden_state
                pooled_embeddings = self._pool_tokens(last_hidden_state, attention_mask)
                all_embeddings.append(pooled_embeddings.cpu().numpy())
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings
        normalized_embeddings = normalize(embeddings)
        
        return normalized_embeddings

def create_philosophical_space(csv_path: str, output_dir: str):
    """Create embeddings from the philosophical excerpts CSV."""
    # Initialize embedder
    embedder = PhilosophicalEmbedder()
    
    # Read CSV
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Processing {len(df)} philosophical excerpts...")
    
    # Generate embeddings for excerpts
    embeddings = embedder.embed_text(df['Excerpt'].tolist())
    
    # Apply PCA reduction
    print("Applying PCA reduction...")
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance ratio with {50} components: {explained_variance:.3f}")
    
    # Create metadata
    metadata = {
        'frameworks': df['Ethical Framework'].unique().tolist(),
        'total_samples': len(df),
        'embedding_dim': reduced_embeddings.shape[1],
        'authors': df['Author'].tolist(),
        'works': df['Work'].tolist(),
        'frameworks_per_sample': df['Ethical Framework'].tolist(),
        'pca_variance_explained': float(explained_variance)
    }
    
    # Save embeddings and metadata
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'philosophical_space.npy'), reduced_embeddings)
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    print(f"Created philosophical space with {len(df)} samples")
    print(f"Embedding dimension: {reduced_embeddings.shape[1]}")
    print(f"Unique frameworks: {len(metadata['frameworks'])}")

if __name__ == "__main__":
    create_philosophical_space(
        csv_path="philosophical_excerpts.csv",
        output_dir="embedded_space"
    )
