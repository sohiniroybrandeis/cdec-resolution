import torch
import numpy
# from torch.utils.data import Dataset
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

train_sentences1 = []
train_sentences2 = []
train_event_triggers1 = []
train_event_triggers2 = []
train_labels = []

dev_sentences1 = []
dev_sentences2 = []
dev_event_triggers1 = []
dev_event_triggers2 = []
dev_labels = []


with open("data 2/event_pairs.train", "r", encoding="utf-8") as file:
    for line in file:
        tokens = line.split('\t')
        train_sentences1.append(tokens[0])
        train_sentences2.append(tokens[11])
        train_event_triggers1.append(int(tokens[1]))
        train_event_triggers2.append(int(tokens[12]))
        train_labels.append(int(tokens[22])) #deal with the other info later

# print(train_sentences)
tr_tokenized = tokenizer(train_sentences1, train_sentences2, padding=True, truncation=True, return_tensors="pt")

with open("data 2/event_pairs.dev", "r", encoding="utf-8") as file:
    for line in file:
        tokens = line.split('\t')
        dev_sentences1.append(tokens[0])
        dev_sentences2.append(tokens[11])
        dev_event_triggers1.append(int(tokens[1]))
        dev_event_triggers2.append(int(tokens[12]))
        dev_labels.append(int(tokens[22])) #deal with the other info later

dev_tokenized = tokenizer(dev_sentences1, dev_sentences2, padding=True, truncation=True, return_tensors="pt")

print(train_labels)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "input_ids": tr_tokenized["input_ids"],
    "attention_mask": tr_tokenized["attention_mask"],
    "trigger_indices_1": torch.tensor(train_event_triggers1),
    "trigger_indices_2": torch.tensor(train_event_triggers2),
    "labels": torch.tensor(train_labels)
})

dev_dataset = Dataset.from_dict({
    "input_ids": dev_tokenized["input_ids"],
    "attention_mask": dev_tokenized["attention_mask"],
    "trigger_indices_1": torch.tensor(dev_event_triggers1),
    "trigger_indices_2": torch.tensor(dev_event_triggers2),
    "labels": torch.tensor(dev_labels)
})

# Define number of labels (e.g., 2 for binary classification)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Change to TaskType.TOKEN_CLS for token classification
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=["query", "value"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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
    eval_dataset=dev_dataset
)

trainer.train()

model.save_pretrained("./roberta_finetuned_lora")

trainer.evaluate()

