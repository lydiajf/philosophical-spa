from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/all-mpnet-base-v2"
# model_name = "allenai/longformer-base-4096"
model = AutoModel.from_pretrained(model_name)
print(model)


def get_model_description():
    model_name = "bert-base-uncased"
    
    print(f"\nLoading {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("\nModel Details:")
    print("-" * 50)
    print(f"Model Name: {model_name}")
    print(f"Number of Parameters: {model.num_parameters():,}")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Number of Layers: {model.config.num_hidden_layers}")
    print(f"Number of Attention Heads: {model.config.num_attention_heads}")
    print(f"Vocabulary Size: {model.config.vocab_size}")
    print(f"Maximum Sequence Length: {model.config.max_position_embeddings}")
    print("-" * 50)
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = get_model_description() 