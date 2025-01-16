from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from typing import List, Dict, Set
import re

def analyze_tokenization(texts: List[str], tokenizer) -> Dict:
    """Analyze how the tokenizer processes philosophical texts."""
    stats = {
        'total_tokens': 0,
        'unknown_tokens': Counter(),  # Track actual unknown tokens
        'token_lengths': [],
        'subword_stats': Counter(),
        'token_frequencies': Counter()  # Track all token frequencies
    }
    
    for text in texts:
        # Get tokens directly
        tokens = tokenizer.tokenize(text)
        stats['total_tokens'] += len(tokens)
        stats['token_lengths'].append(len(tokens))
        
        # Track token frequencies and unknown tokens
        for token in tokens:
            stats['token_frequencies'][token] += 1
            if token == tokenizer.unk_token:
                # Try to find what caused the unknown token by looking at small windows
                window = 5
                words = text.split()
                for i in range(len(words) - window + 1):
                    window_text = ' '.join(words[i:i+window])
                    window_tokens = tokenizer.tokenize(window_text)
                    if tokenizer.unk_token in window_tokens:
                        stats['unknown_tokens'][window_text] += 1
        
        # Analyze multi-token words (subword tokenization)
        words = text.split()
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 1:
                stats['subword_stats'][word] = len(word_tokens)
    
    return stats

def plot_tokenization_stats(stats: Dict):
    """Plot various statistics about tokenization."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Token length distribution
    plt.subplot(2, 2, 1)
    plt.hist(stats['token_lengths'], bins=30)
    plt.title('Distribution of Text Lengths (in tokens)')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    # Plot 2: Text segments producing unknown tokens
    plt.subplot(2, 2, 2)
    unknown_tokens = list(stats['unknown_tokens'].most_common(10))
    if unknown_tokens:
        segments, counts = zip(*unknown_tokens)
        plt.barh([s[:20] + '...' if len(s) > 20 else s for s in segments], counts)
        plt.title('Text Segments Producing Unknown Tokens')
        plt.xlabel('Frequency')
    
    # Plot 3: Most common tokens
    plt.subplot(2, 2, 3)
    common_tokens = list(stats['token_frequencies'].most_common(10))
    if common_tokens:
        tokens, counts = zip(*common_tokens)
        plt.barh(tokens, counts)
        plt.title('Most Common Tokens')
        plt.xlabel('Frequency')
    
    # Plot 4: Words split into most subwords
    plt.subplot(2, 2, 4)
    subword_tokens = list(stats['subword_stats'].most_common(10))
    if subword_tokens:
        words, counts = zip(*subword_tokens)
        plt.barh([w[:20] + '...' if len(w) > 20 else w for w in words], counts)
        plt.title('Words Split into Most Subwords')
        plt.xlabel('Number of Subwords')
    
    plt.tight_layout()
    plt.show()

def test_philosophical_tokenizer(csv_path: str):
    """Test tokenizer's handling of philosophical texts."""
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    texts = df['Excerpt'].tolist()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    # Basic tokenizer info
    print(f"\nTokenizer Info:")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    # Analyze tokenization
    print("\nAnalyzing tokenization...")
    stats = analyze_tokenization(texts, tokenizer)
    
    # Print statistics
    print(f"\nTokenization Statistics:")
    print(f"Total tokens processed: {stats['total_tokens']}")
    print(f"Number of text segments producing unknown tokens: {len(stats['unknown_tokens'])}")
    print(f"Average tokens per text: {np.mean(stats['token_lengths']):.2f}")
    
    # Print text segments producing unknown tokens
    print("\nText segments producing unknown tokens:")
    for segment, count in stats['unknown_tokens'].most_common(10):
        print(f"'{segment}': {count} occurrences")
    
    # Print most common tokens
    print("\nMost common tokens:")
    for token, count in stats['token_frequencies'].most_common(20):
        print(f"'{token}': {count} occurrences")
    
    # Print words split into most subwords
    print("\nWords split into most subwords:")
    for word, num_subwords in stats['subword_stats'].most_common(10):
        subwords = tokenizer.tokenize(word)
        print(f"'{word}' -> {subwords} ({num_subwords} subwords)")
    
    # Plot statistics
    plot_tokenization_stats(stats)

if __name__ == "__main__":
    test_philosophical_tokenizer("philosophical_excerpts.csv")
