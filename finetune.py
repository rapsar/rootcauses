import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_contrastive_pairs(jsonl_file):
    """Load contrastive pairs from JSONL file"""
    pairs = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            pairs.append({
                'text1': data['quantity'],
                'text2': data['concept'], 
                'label': 1 if data['related_to'] else 0
            })
    return pairs

def create_input_examples(pairs):
    """Convert pairs to SentenceTransformers InputExample format"""
    examples = []
    for pair in pairs:
        examples.append(InputExample(
            texts=[pair['text1'], pair['text2']], 
            label=float(pair['label'])
        ))
    return examples

def main():
    # Configuration
    JSONL_FILE = "data/physics_pairs.jsonl"  # Your input file
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    OUTPUT_DIR = f"physics-embedding-model-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Training parameters
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    EVAL_STEPS = 500
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load the base model
    logging.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Load contrastive pairs
    logging.info(f"Loading pairs from: {JSONL_FILE}")
    pairs = load_contrastive_pairs(JSONL_FILE)
    logging.info(f"Loaded {len(pairs)} pairs")
    
    # Count positive/negative pairs
    pos_pairs = sum(1 for p in pairs if p['label'] == 1)
    neg_pairs = len(pairs) - pos_pairs
    logging.info(f"Positive pairs: {pos_pairs}, Negative pairs: {neg_pairs}")
    
    # Split into train/validation
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42, 
                                            stratify=[p['label'] for p in pairs])
    
    # Convert to InputExample format
    train_examples = create_input_examples(train_pairs)
    val_examples = create_input_examples(val_pairs)
    
    logging.info(f"Training examples: {len(train_examples)}")
    logging.info(f"Validation examples: {len(val_examples)}")
    
    # Create data loaders
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    
    # Define loss function - ContrastiveLoss works well for binary classification
    train_loss = losses.ContrastiveLoss(model=model)
    
    # Alternative loss functions you could try:
    # train_loss = losses.CosineSimilarityLoss(model=model)
    # train_loss = losses.OnlineContrastiveLoss(model=model)
    
    # Create evaluator for validation
    val_texts1 = [ex.texts[0] for ex in val_examples]
    val_texts2 = [ex.texts[1] for ex in val_examples] 
    val_labels = [ex.label for ex in val_examples]
    
    evaluator = BinaryClassificationEvaluator(
        sentences1=val_texts1,
        sentences2=val_texts2, 
        labels=val_labels,
        batch_size=EVAL_BATCH_SIZE,
        name="physics_validation"
    )
    
    # Training arguments
    warmup_steps = min(WARMUP_STEPS, len(train_dataloader) * NUM_EPOCHS // 10)
    
    logging.info("Starting training...")
    logging.info(f"Training for {NUM_EPOCHS} epochs")
    logging.info(f"Batch size: {TRAIN_BATCH_SIZE}")
    logging.info(f"Learning rate: {LEARNING_RATE}")
    logging.info(f"Warmup steps: {warmup_steps}")
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=EVAL_STEPS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        optimizer_params={'lr': LEARNING_RATE},
        use_amp=True,  # Enable automatic mixed precision for GPU speedup
        save_best_model=True
    )
    
    logging.info(f"Training completed! Model saved to: {OUTPUT_DIR}")
    
    # Test the fine-tuned model
    logging.info("Testing the fine-tuned model...")
    test_pairs = [
        ("Temperature", "Kelvin"),
        ("Force", "Newton's second law"), 
        ("Quantum tunneling", "Classical mechanics"),  # Should be dissimilar
        ("Energy", "Kinetic energy")
    ]
    
    for text1, text2 in test_pairs:
        embedding1 = model.encode([text1])
        embedding2 = model.encode([text2])
        similarity = torch.cosine_similarity(torch.tensor(embedding1), 
                                           torch.tensor(embedding2), dim=1)
        logging.info(f"Similarity between '{text1}' and '{text2}': {similarity.item():.4f}")

if __name__ == "__main__":
    main()

# Additional utility functions for testing and evaluation

def evaluate_model_on_physics_concepts(model_path, test_jsonl=None):
    """Evaluate the fine-tuned model on physics concept pairs"""
    model = SentenceTransformer(model_path)
    
    if test_jsonl:
        test_pairs = load_contrastive_pairs(test_jsonl)
        
        correct = 0
        total = len(test_pairs)
        threshold = 0.5  # Adjust based on your needs
        
        for pair in test_pairs:
            emb1 = model.encode([pair['text1']])
            emb2 = model.encode([pair['text2']])
            similarity = torch.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2), dim=1).item()
            
            predicted = 1 if similarity > threshold else 0
            if predicted == pair['label']:
                correct += 1
        
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy

def compare_models(base_model_name, finetuned_model_path, test_concepts):
    """Compare base model vs fine-tuned model on physics concepts"""
    base_model = SentenceTransformer(base_model_name)
    finetuned_model = SentenceTransformer(finetuned_model_path)
    
    print("Model Comparison:")
    print("================")
    
    for text1, text2 in test_concepts:
        # Base model
        base_emb1 = base_model.encode([text1])
        base_emb2 = base_model.encode([text2])
        base_sim = torch.cosine_similarity(torch.tensor(base_emb1), torch.tensor(base_emb2), dim=1).item()
        
        # Fine-tuned model  
        ft_emb1 = finetuned_model.encode([text1])
        ft_emb2 = finetuned_model.encode([text2])
        ft_sim = torch.cosine_similarity(torch.tensor(ft_emb1), torch.tensor(ft_emb2), dim=1).item()
        
        print(f"'{text1}' <-> '{text2}':")
        print(f"  Base model: {base_sim:.4f}")
        print(f"  Fine-tuned: {ft_sim:.4f}")
        print(f"  Difference: {ft_sim - base_sim:+.4f}")
        print()

# Example usage after training:
# evaluate_model_on_physics_concepts("./physics-embedding-model-20241201_143022/")
# 
# test_concepts = [
#     ("Force", "Newton's second law"),
#     ("Temperature", "Kelvin"), 
#     ("Quantum mechanics", "Wave function"),
#     ("Classical mechanics", "Quantum tunneling")  # Should be less similar
# ]
# compare_models("sentence-transformers/all-MiniLM-L6-v2", 
#                "./physics-embedding-model-20241201_143022/", 
#                test_concepts)