import torch
import numpy
# from torch.utils.data import Dataset
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

train_sentences = []
train_event_triggers = []
train_labels = []

dev_sentences = []
dev_event_triggers = []
dev_labels = []

# Example labels (modify for your task)
labels = torch.tensor([0, 1])  # Adjust label format for your use case

with open("/content/data2/event_pairs.train", "r", encoding="utf-8") as file:
    for line in file:
        tokens = line.split('\t')
        train_sentences.append(tokens[0])
        train_sentences.append(tokens[11])
        train_event_triggers.append(int(tokens[1]))
        train_event_triggers.append(int(tokens[2]))
        train_labels.append(int(tokens[10])) #deal with the other info later

tr_tokenized = tokenizer(train_sentences, return_offsets_mapping=True, padding=True, truncation=True, return_tensors="pt")

with open("/content/data2/event_pairs.dev", "r", encoding="utf-8") as file:
    for line in file:
        tokens = line.split('\t')
        dev_sentences.append(tokens[0])
        dev_sentences.append(tokens[11])
        dev_event_triggers.append(tokens[1])
        dev_event_triggers.append(tokens[2])
        dev_labels.append(tokens[10]) #deal with the other info later

dev_tokenized = tokenizer(dev_sentences, return_offsets_mapping=True, padding=True, truncation=True, return_tensors="pt")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "input_ids": tr_tokenized["input_ids"],
    "attention_mask": tr_tokenized["attention_mask"],
    "trigger_indices": torch.tensor(train_event_triggers),
    "labels": labels
})

dev_dataset = Dataset.from_dict({
    "input_ids": dev_tokenized["input_ids"],
    "attention_mask": dev_tokenized["attention_mask"],
    "trigger_indices": torch.tensor(dev_event_triggers),
    "labels": labels
})

# Define number of labels (e.g., 2 for binary classification)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./roberta_finetuned",  # Where to save model
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Your training data
    dev_dataset=dev_dataset
)

trainer.train()

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Change to TaskType.TOKEN_CLS for token classification
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.save_pretrained("./roberta_finetuned_lora")

trainer.evaluate()
