# Embed Me: A Philosophical Semantic Space

## Overview
**Embed Me** is an experimental philosophy project that uses machine learning to map and structure philosophical thought. Using high-dimensional embedding models, the aim is to capture the conceptual landscape of philosophy, helping users locate their intellectual influences and philosophical contrasts.

## Features
- **Philosophical Embedding Space:** Generates high-dimensional vector representations of philosophical texts.
- **Personalized Semantic Mapping:** Users can embed their own writing and discover their nearest and farthest philosophical neighbors.
- **Text Similarity Analysis:** Uses cosine similarity and nearest-neighbor techniques to measure conceptual proximity.
- **Philosophical Recommender:** Suggests relevant thinkers, books, and intellectual influences based on user input.
- **Model Evaluation & Fine-Tuning:** Initial testing with a BERT model, with potential future fine-tuning of LLMs to improve nuance.

## Tech Stack
- **Embedding Models:** BERT, next step using LLM prompting to force defined spaces.
- **Data Sources:** Excerpts from classic philosophical texts (e.g., Project Gutenberg, curated book lists).
- **Evaluation Metrics:** Cosine similarity, nearest-neighbor searches, and reduction techniques (e.g., PhilPapers' 126D space).

## Next Steps
- **Improving Embeddings:** Enhance model precision by increasing distance between embeddings with prior prompting and possibly fine-tuning using philosophical-specific datasets. Also need to consider relationships between philosophers and texts.
- **Dynamic User Input:** Allow users to iteratively adjust and refine their semantic space.
- **Visual Representation:** Develop an interactive tool to explore conceptual relationships.
