import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import numpy as np
import os
import json
import pandas as pd

class PhilosophicalEmbedder:
    def __init__(
        self, 
        model_name: str = "allenai/longformer-base-4096",
        layers_to_combine: List[int] = [-4, -3, -2, -1],  # Last 4 layers by default
        pooling_strategy: str = "mean"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.layers_to_combine = layers_to_combine
        self.pooling_strategy = pooling_strategy
        
    def _combine_layers(self, hidden_states: tuple) -> torch.Tensor:
        """Combine multiple layers of embeddings."""
        selected_layers = [hidden_states[i] for i in self.layers_to_combine]
        combined = torch.stack(selected_layers)
        return torch.mean(combined, dim=0)
    
    def _pool_tokens(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings into a single vector."""
        if self.pooling_strategy == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_strategy == "cls":
            return token_embeddings[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text(s)."""
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
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
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            combined_layers = self._combine_layers(outputs.hidden_states)
            pooled_embeddings = self._pool_tokens(combined_layers, attention_mask)
            
        return pooled_embeddings.cpu().numpy()

def create_philosophical_space(csv_path: str, output_dir: str):
    """Create embeddings from the philosophical excerpts CSV."""
    # Initialize embedder
    embedder = PhilosophicalEmbedder()
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print(f"Processing {len(df)} philosophical excerpts...")
    
    # Generate embeddings for excerpts
    embeddings = embedder.embed_text(df['Excerpt'].tolist())
    
    # Create metadata
    metadata = {
        'frameworks': df['Ethical Framework'].unique().tolist(),
        'total_samples': len(df),
        'embedding_dim': embeddings.shape[1],
        'authors': df['Author'].tolist(),
        'works': df['Work'].tolist(),
        'frameworks_per_sample': df['Ethical Framework'].tolist()
    }
    
    # Save embeddings and metadata
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'philosophical_space.npy'), embeddings)
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    print(f"Created philosophical space with {len(df)} samples")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Unique frameworks: {len(metadata['frameworks'])}")

if __name__ == "__main__":
    create_philosophical_space(
        csv_path="philosophical_excerpts.csv",
        output_dir="embedded_space"
    )
