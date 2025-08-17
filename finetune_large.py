import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator, InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging
import os
import glob
import random
from datetime import datetime
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def load_physics_qa_data(jsonl_file: str) -> List[Dict]:
   """Load physics Q&A data from a single JSONL file"""
   qa_pairs = []
   
   logging.info(f"Loading data from JSONL file: {jsonl_file}")
   
   try:
       with open(jsonl_file, 'r', encoding='utf-8') as f:
           for line_num, line in enumerate(f, 1):
               try:
                   data = json.loads(line.strip())
                   
                   # Extract question and answer
                   question = data.get('message_1', '').strip()
                   answer = data.get('message_2', '').strip()
                   topic = data.get('topic', '').strip()
                   sub_topic = data.get('sub_topic', '').strip()
                   
                   if question and answer:
                       qa_pairs.append({
                           'question': question,
                           'answer': answer,
                           'topic': topic,
                           'sub_topic': sub_topic,
                           'line_num': line_num
                       })
               except json.JSONDecodeError as e:
                   logging.warning(f"Error parsing JSON at line {line_num}: {e}")
               except KeyError as e:
                   logging.warning(f"Missing key at line {line_num}: {e}")
   
   except FileNotFoundError:
       logging.error(f"File not found: {jsonl_file}")
       return []
   except Exception as e:
       logging.error(f"Error reading file {jsonl_file}: {e}")
       return []
   
   logging.info(f"Loaded {len(qa_pairs)} Q&A pairs from JSONL file")
   return qa_pairs

def create_contrastive_examples(qa_pairs: List[Dict], neg_ratio: float = 1.0) -> List[InputExample]:
    """
    Create contrastive examples from Q&A pairs for embedding training
    
    Positive examples: (question, correct_answer)
    Negative examples: (question, random_wrong_answer)
    """
    examples = []
    
    # Create positive examples (question paired with its correct answer)
    for pair in qa_pairs:
        examples.append(InputExample(
            texts=[pair['question'], pair['answer']], 
            label=1.0
        ))
    
    # Create negative examples (question paired with random wrong answer)
    num_negatives = int(len(qa_pairs) * neg_ratio)
    for i in range(num_negatives):
        # Pick a random question
        question_idx = random.randint(0, len(qa_pairs) - 1)
        question = qa_pairs[question_idx]['question']
        
        # Pick a random different answer
        answer_idx = random.randint(0, len(qa_pairs) - 1)
        while answer_idx == question_idx:  # Ensure it's a different answer
            answer_idx = random.randint(0, len(qa_pairs) - 1)
        
        wrong_answer = qa_pairs[answer_idx]['answer']
        
        examples.append(InputExample(
            texts=[question, wrong_answer],
            label=0.0
        ))
    
    # Shuffle the examples
    random.shuffle(examples)
    
    logging.info(f"Created {len(examples)} training examples ({len(qa_pairs)} positive, {num_negatives} negative)")
    return examples

def create_topic_based_examples(qa_pairs: List[Dict]) -> List[InputExample]:
    """
    Create examples based on topic similarity
    - Questions from same topic/sub_topic get higher similarity
    - Questions from different topics get lower similarity
    """
    examples = []
    
    # Group by topic and sub_topic
    topic_groups = {}
    for i, pair in enumerate(qa_pairs):
        topic_key = f"{pair['topic']}|{pair['sub_topic']}"
        if topic_key not in topic_groups:
            topic_groups[topic_key] = []
        topic_groups[topic_key].append((i, pair))
    
    # Create positive examples within same topic/sub_topic
    for topic_key, pairs_in_topic in topic_groups.items():
        if len(pairs_in_topic) > 1:
            for i in range(len(pairs_in_topic)):
                for j in range(i + 1, min(i + 3, len(pairs_in_topic))):  # Limit pairs per topic
                    pair1 = pairs_in_topic[i][1]
                    pair2 = pairs_in_topic[j][1]
                    
                    examples.append(InputExample(
                        texts=[pair1['question'], pair2['question']],
                        label=0.8  # High similarity for same sub-topic
                    ))
    
    # Create medium similarity examples for same main topic, different sub-topic
    main_topics = {}
    for pair in qa_pairs:
        main_topic = pair['topic']
        if main_topic not in main_topics:
            main_topics[main_topic] = []
        main_topics[main_topic].append(pair)
    
    for main_topic, pairs_in_main_topic in main_topics.items():
        if len(pairs_in_main_topic) > 5:  # Only if we have enough examples
            sample_size = min(10, len(pairs_in_main_topic))
            sampled_pairs = random.sample(pairs_in_main_topic, sample_size)
            
            for i in range(len(sampled_pairs)):
                for j in range(i + 1, len(sampled_pairs)):
                    pair1, pair2 = sampled_pairs[i], sampled_pairs[j]
                    if pair1['sub_topic'] != pair2['sub_topic']:  # Different sub-topics
                        examples.append(InputExample(
                            texts=[pair1['question'], pair2['question']],
                            label=0.5  # Medium similarity for same main topic
                        ))
    
    # Create negative examples for different main topics
    main_topic_list = list(main_topics.keys())
    if len(main_topic_list) > 1:
        for _ in range(len(qa_pairs) // 2):  # Create some negative examples
            topic1, topic2 = random.sample(main_topic_list, 2)
            pair1 = random.choice(main_topics[topic1])
            pair2 = random.choice(main_topics[topic2])
            
            examples.append(InputExample(
                texts=[pair1['question'], pair2['question']],
                label=0.1  # Low similarity for different main topics
            ))
    
    logging.info(f"Created {len(examples)} topic-based examples")
    return examples

def create_evaluation_data(qa_pairs: List[Dict], eval_size: int = 200) -> Tuple[List[str], List[str], List[int]]:
    """Create evaluation data for BinaryClassificationEvaluator"""
    eval_pairs = random.sample(qa_pairs, min(eval_size, len(qa_pairs)))
    
    sentences1, sentences2, labels = [], [], []
    
    # Positive examples (question with correct answer)
    for pair in eval_pairs[:eval_size//2]:
        sentences1.append(pair['question'])
        sentences2.append(pair['answer'])
        labels.append(1)
    
    # Negative examples (question with wrong answer)
    for i in range(eval_size//2):
        question_idx = i
        answer_idx = (i + len(eval_pairs)//4) % len(eval_pairs)  # Different answer
        
        sentences1.append(eval_pairs[question_idx]['question'])
        sentences2.append(eval_pairs[answer_idx]['answer'])
        labels.append(0)
    
    return sentences1, sentences2, labels

def main():
    # Configuration
    DATA_DIR = "physics_dataset"  # Directory containing JSON files
    JSONL_FILE = "physics_combined.jsonl"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    OUTPUT_DIR = f"physics-embedding-model-large-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Training parameters - optimized for A4000
    TRAIN_BATCH_SIZE = 16  # Reduced for A4000 memory
    EVAL_BATCH_SIZE = 32
    NUM_EPOCHS = 3  # Reduced for large dataset
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 500
    EVAL_STEPS = 100
    MAX_SEQ_LENGTH = 256  # Reduced to fit more in memory
    
    # Data parameters
    NEGATIVE_RATIO = 0.8  # Ratio of negative to positive examples
    USE_TOPIC_BASED = True  # Whether to include topic-based examples
    
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load the base model
    logging.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    
    # Load physics Q&A data
    logging.info(f"Loading data from: {DATA_DIR}")
    qa_pairs = load_physics_qa_data(os.path.join(DATA_DIR, JSONL_FILE))
    qa_pairs = qa_pairs[:4000]
    
    if len(qa_pairs) == 0:
        logging.error("No data loaded. Please check the data directory and file format.")
        return
    
    # Display data statistics
    topics = set(pair['topic'] for pair in qa_pairs)
    logging.info(f"Found {len(topics)} unique topics: {sorted(topics)}")
    
    # Create training examples
    logging.info("Creating contrastive examples...")
    examples = create_contrastive_examples(qa_pairs, neg_ratio=NEGATIVE_RATIO)
    
    if USE_TOPIC_BASED:
        logging.info("Creating topic-based examples...")
        topic_examples = create_topic_based_examples(qa_pairs)
        examples.extend(topic_examples)
        random.shuffle(examples)
    
    # Split into train/validation
    train_examples, val_examples = train_test_split(
        examples, test_size=0.15, random_state=42, 
        stratify=[1 if ex.label > 0.5 else 0 for ex in examples]
    )
    
    logging.info(f"Training examples: {len(train_examples)}")
    logging.info(f"Validation examples: {len(val_examples)}")
    
    # Create data loaders
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    # Alternative loss functions:
    # train_loss = losses.ContrastiveLoss(model=model)
    # train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Create evaluator for validation
    eval_sentences1, eval_sentences2, eval_labels = create_evaluation_data(qa_pairs)
    
    evaluator = BinaryClassificationEvaluator(
        sentences1=eval_sentences1,
        sentences2=eval_sentences2,
        labels=eval_labels,
        batch_size=EVAL_BATCH_SIZE,
        name="physics_qa_validation"
    )
    
    # Training arguments
    warmup_steps = min(WARMUP_STEPS, len(train_dataloader) * NUM_EPOCHS // 10)
    
    logging.info("Starting training...")
    logging.info(f"Training for {NUM_EPOCHS} epochs")
    logging.info(f"Batch size: {TRAIN_BATCH_SIZE}")
    logging.info(f"Learning rate: {LEARNING_RATE}")
    logging.info(f"Warmup steps: {warmup_steps}")
    logging.info(f"Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=EVAL_STEPS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        optimizer_params={'lr': LEARNING_RATE},
        use_amp=True,  # Enable automatic mixed precision for memory efficiency
        save_best_model=True,
        checkpoint_save_steps=5000,  # Save checkpoints
        checkpoint_save_total_limit=2
    )
    
    logging.info(f"Training completed! Model saved to: {OUTPUT_DIR}")
    
    # Test the fine-tuned model
    logging.info("Testing the fine-tuned model...")
    test_physics_qa(model, qa_pairs[:5])  # Test on first 5 examples

def test_physics_qa(model, qa_pairs: List[Dict]):
    """Test the model on some Q&A pairs"""
    logging.info("Testing model performance on sample Q&A pairs:")
    
    for i, pair in enumerate(qa_pairs):
        question = pair['question']
        correct_answer = pair['answer']
        
        # Get embeddings
        q_embedding = model.encode([question])
        a_embedding = model.encode([correct_answer])
        
        # Calculate similarity
        similarity = torch.cosine_similarity(
            torch.tensor(q_embedding), 
            torch.tensor(a_embedding), 
            dim=1
        ).item()
        
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Topic: {pair['topic']} -> {pair['sub_topic']}")
        logging.info(f"Question: {question[:100]}...")
        logging.info(f"Q-A Similarity: {similarity:.4f}")

def evaluate_retrieval_performance(model_path: str, qa_pairs: List[Dict], sample_size: int = 500):
    """Evaluate the model's retrieval performance"""
    model = SentenceTransformer(model_path)
    
    # Sample some Q&A pairs for evaluation
    eval_pairs = random.sample(qa_pairs, min(sample_size, len(qa_pairs)))
    
    # Create corpus of answers
    corpus = [pair['answer'] for pair in eval_pairs]
    queries = [pair['question'] for pair in eval_pairs]
    
    # Create relevance mapping (each question is relevant to its own answer)
    relevant_docs = {i: {i} for i in range(len(queries))}
    
    # Use InformationRetrievalEvaluator
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="physics_qa_retrieval"
    )
    
    results = ir_evaluator(model)
    print("Retrieval Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

# Example usage after training:
if __name__ == "__main__":
    main()
    
    # Uncomment to run additional evaluations after training
    # qa_data = load_physics_qa_data("physics")
    # evaluate_retrieval_performance("./physics-embedding-model-YYYYMMDD_HHMMSS/", qa_data)