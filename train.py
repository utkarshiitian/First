import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import argparse
from sklearn.model_selection import train_test_split
import torch
import os
import warnings
warnings.filterwarnings('ignore')

def main(args):
    """
    Main function to run the model training and evaluation pipeline.
    """
    print("--- Starting Model Training ---")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Prepare Data
    print(f"Loading dataset from {args.train_csv}...")
    try:
        df = pd.read_csv(args.train_csv)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {args.train_csv} not found!")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Check required columns
    required_columns = ['origin_query', 'cate_path', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Handle missing values
    print("Checking for missing values...")
    if df[required_columns].isnull().any().any():
        print("Warning: Found missing values. Dropping rows with NaN...")
        df = df.dropna(subset=required_columns)
        print(f"Dataset shape after dropping NaN: {df.shape}")
    
    # Create text input
    df['text_input'] = df['origin_query'].astype(str) + " [SEP] " + df['cate_path'].astype(str)
    print("Data prepared with 'text_input' column.")
    
    # Ensure labels are integers
    df['label'] = df['label'].astype(int)
    
    # Check label distribution
    print("\nLabel distribution:")
    print(df['label'].value_counts())

    # Split data into training and validation sets (90% train, 10% validation)
    try:
        train_df, val_df = train_test_split(
            df, 
            test_size=0.1, 
            random_state=42, 
            stratify=df['label']
        )
        print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
    except ValueError as e:
        print(f"Error in train_test_split: {e}")
        print("Using random split without stratification...")
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Reset index to avoid issues with Dataset.from_pandas
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df[['text_input', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text_input', 'label']])
    
    # Combine into a DatasetDict for the Trainer
    ds = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    # 2. Load Tokenizer and Model
    print(f"\nLoading pre-trained model and tokenizer: {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, 
            num_labels=2,
            ignore_mismatched_sizes=True  # Add this to handle size mismatches
        )
        # Move model to device
        model = model.to(device)
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return

    # 3. Tokenize Dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text_input"], 
            padding="max_length", 
            truncation=True,
            max_length=512  # Explicitly set max length
        )

    print("\nTokenizing datasets...")
    try:
        tokenized_ds = ds.map(tokenize_function, batched=True)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        print("Tokenization complete!")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return

    # 4. Define Training Arguments
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Specify metric for best model
        greater_is_better=False,        # Lower loss is better
        save_total_limit=2,             # Keep only 2 best checkpoints
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
        report_to="none",               # Disable wandb/tensorboard reporting
        push_to_hub=False,              # Don't push to hub
    )

    # 5. Initialize and Run Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        tokenizer=tokenizer,  # Add tokenizer for better checkpoint saving
    )

    print("\n--- Starting fine-tuning ---")
    try:
        trainer.train()
        print("--- Fine-tuning complete ---")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # 6. Save the final best model and tokenizer
    print(f"\nSaving final model to {args.model_save_path}...")
    try:
        os.makedirs(args.model_save_path, exist_ok=True)
        trainer.save_model(args.model_save_path)
        tokenizer.save_pretrained(args.model_save_path)
        print("Model saved successfully!")
        
        # Print final validation metrics
        eval_results = trainer.evaluate()
        print(f"\nFinal validation loss: {eval_results['eval_loss']:.4f}")
        
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a multilingual model for query-category classification.")
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file.')
    parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased', help='Name of the pre-trained model.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory for checkpoints.')
    parser.add_argument('--model_save_path', type=str, default='./finetuned_classifier', help='Directory to save the final model.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    
    args = parser.parse_args()
    main(args)
