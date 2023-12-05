# WEEK 9 HW - JaMor Hairston

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, f1_score
import torch

# 1a. Build a text classification model using a transformer-based architecture using the airline sentiment dataset. 
    # Split the dataset into the training and test sets, 80% for training and 20% for testing.

# Load the dataset
data = pd.read_csv('/Provided Files/Tweets.csv')
texts = data['text']
labels = data['airline_sentiment']
ids = data['tweet_id']

# Split the dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer and encode the texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Convert labels to tensors
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=test_encodings,
)

# Train the model
trainer.train()

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(**test_encodings)
    predicted_labels = predictions.logits.argmax(dim=1)

# Calculate evaluation metrics
overall_f1 = f1_score(test_labels, predicted_labels, average='micro')
report = classification_report(test_labels, predicted_labels, target_names=['negative', 'neutral', 'positive'])

# Print evaluation results
print(f"Overall Micro-weighted F1-score: {overall_f1}")
print("Classification Report:")
print(report)


'''
	Cited Sources: 
    1. Stack Overflow and ChatGPT used for debugging with model training.
'''

