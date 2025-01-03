import os
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

CUDA_LAUNCH_BLOCKING=1

# Define paths and model
model_name_or_path = "dmis-lab/biobert-base-cased-v1.1"
train_file = "./scripts/lara_preprocessed_conll/train.conll"
dev_file = "./scripts/lara_preprocessed_conll/dev.conll"
test_file = "./scripts/lara_preprocessed_conll/test.conll"
num_labels = 3  
label_list = ["O", "B-PK", "I-PK"] 

# Label encoding (map string labels to integer labels)
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, 
                                                       num_labels=num_labels,
                                                       label2id=label2id, 
                                                       id2label=id2label)

# Preprocess CoNLL-formatted data into tokenized format
def preprocess_conll_data(file_path):
    """
    Preprocess CoNLL-formatted data into tokenized format for NER tasks.
    Args:
    - file_path: Path to the CoNLL data file.
    Returns:
    - A dictionary containing 'text' and 'labels' for training.
    """
    texts = []
    labels = []
    current_text = []
    current_labels = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":  # Empty line indicates a sentence break
                if current_text:
                    texts.append(current_text)
                    labels.append(current_labels)
                    current_text = []
                    current_labels = []
            else:
                token, label = line.strip().split()  # Split token and label
                current_text.append(token)
                current_labels.append(label)
        
        # Add the last sentence if not empty
        if current_text:
            texts.append(current_text)
            labels.append(current_labels)
    
    return {"text": texts, "labels": labels}

# Load and preprocess datasets
train_data = preprocess_conll_data(train_file)
dev_data = preprocess_conll_data(dev_file)
test_data = preprocess_conll_data(test_file)

# Convert preprocessed data to HuggingFace dataset format
from datasets import Dataset
train_dataset = Dataset.from_dict(train_data)
dev_dataset = Dataset.from_dict(dev_data)
test_dataset = Dataset.from_dict(test_data)

# Align labels with tokens
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        is_split_into_words=True,  # Ensure the tokenizer understands words are pre-split
    )

    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their word IDs
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:  # Special tokens (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Start of a new word
                label_ids.append(label2id[labels[word_idx]])  # Map label to integer
            else:  # Continuation of the same word
                #label_ids.append(label2id[labels[word_idx]])  # Map label to integer
                label_ids.append(-100) 
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Apply tokenization and alignment to all datasets
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# Load metric
metric = load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Compute class weights (class_weight='balanced' adjusts for imbalance)
#all_labels = [label for sentence_labels in train_data['labels'] for label in sentence_labels]

all_labels = [label2id[label] for sentence_labels in train_data['labels'] for label in sentence_labels]

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(all_labels), y=all_labels)
class_weights_dict = {i: weight for i, weight in zip(np.unique(all_labels), class_weights)}

print("Class weights (numeric):", class_weights_dict)

# Create a mapping from label to weight
class_weights_dict = {label2id[label]: weight for label, weight in zip(label_list, class_weights)}

print("Class weights:", class_weights_dict)


#Attempting Implementation of Focal Loss to Less Aggressively weight the classes

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        """
        Focal Loss for class imbalance.

        Args:
        - gamma (float): Focusing parameter to reduce the loss for well-classified examples.
        - alpha (list, optional): Class weights to address class imbalance (can be set to `None`).
        - ignore_index (int): Index to ignore in loss computation.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Forward pass for Focal Loss.

        Args:
        - logits (Tensor): Model predictions of shape (batch_size, num_classes).
        - targets (Tensor): Ground-truth labels of shape (batch_size).

        Returns:
        - loss (Tensor): Computed Focal Loss.
        """
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1), 
            reduction="none", 
            ignore_index=self.ignore_index
        )

        # Get the probabilities for the true class
        pt = torch.exp(-ce_loss)

        # Apply the focal loss scaling factor
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply class weights (if provided)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss

        # Compute the mean loss
        return focal_loss.mean()



# -------------- **NEW**: Define custom loss function to apply class weights -------------------
#class CustomTrainer(Trainer):
 #   def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
  #      """
  #      Override the Trainer's compute_loss method to apply class weights and handle additional arguments.
   #     """
    #    #labels = inputs.get("labels").long()
     #   labels = inputs["labels"].to(dtype=torch.long, device=model.device)
      #  outputs = model(**inputs)
       ## logits = model(**inputs).logits.float()
       # logits = outputs.logits.to(dtype=torch.float32)

        # Forward pass
        #outputs = model(**inputs)
       # logits = outputs.logits

        
        # Compute the loss using class weights
       # loss_fct = CrossEntropyLoss(weight=torch.tensor(list(class_weights_dict.values())).to(inputs['input_ids'].device))
       # loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
       # loss_fct = CrossEntropyLoss(
        #    weight=torch.tensor(list(class_weights_dict.values())).to(labels.device)
        #)
        #loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
     #   weights = torch.tensor(list(class_weights_dict.values()), dtype=torch.float32).to(labels.device)
      #  loss_fct = CrossEntropyLoss(weight=weights)
        
       # print(f"logits shape: {logits.shape}, dtype: {logits.dtype}")
      #  print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")

        # Handle any additional arguments passed to the method
        #return (loss, logits) if return_outputs else loss
      #  loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
       # return (loss, outputs) if return_outputs else loss

#Custom Trainer with Focal Loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override the Trainer's compute_loss method to apply Focal Loss.
        """
        labels = inputs["labels"].to(dtype=torch.long, device=model.device)
        outputs = model(**inputs)
        logits = outputs.logits

        # Define Focal Loss
        weights = torch.tensor(list(class_weights_dict.values()), dtype=torch.float32).to(labels.device)
        loss_fct = FocalLoss(gamma=2.0, alpha=weights)

        # Compute the loss
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Training arguments with early stopping
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  
    greater_is_better=True,
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=10  # Stop training if no improvement after 3 evaluations
)

# Initialize Trainer with custom loss function
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./fine_tuned_biobert_ner")
tokenizer.save_pretrained("./fine_tuned_biobert_ner")

# Evaluate on test dataset
test_results = trainer.evaluate(test_dataset)

print("Test Results:")
for key, value in test_results.items():
    print(f"{key}: {value}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inference on new text using fine-tuned model
ner_pipeline = pipeline("ner", model="./fine_tuned_biobert_ner", tokenizer="./fine_tuned_biobert_ner", device=0 if torch.cuda.is_available() else -1)
sample_text = "The pharmacokinetic profile of the drug revealed that the maximum plasma concentration (Cmax) was reached 2 hours post-administration, with a half-life of approximately 6 hours, indicating the need for twice-daily dosing to maintain therapeutic levels, while the area under the curve (AUC) suggested sufficient bioavailability for effective treatment of the condition."
results = ner_pipeline(sample_text)

print("\nNER Results on Sample Text:")
for result in results:
    print(result)
