import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base") # using pre-trained roberta-base tokenizer

def load_data(file_path):
    sentences1 = []
    sentences2 = []
    event_triggers1 = []
    event_triggers2 = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            tokens = line.split('\t')
            sentences1.append(tokens[0])
            sentences2.append(tokens[11])
            event_triggers1.append(int(tokens[1]))
            event_triggers2.append(int(tokens[12]))
            labels.append(int(tokens[22])) # recognizing info based on tab-separation, adding to lists

    tokenized = tokenizer(sentences1, sentences2, padding=True, truncation=True, return_tensors="pt")

    dataset = Dataset.from_dict({ # convert to Hugging Face Dataset
    "input_ids": tokenized["input_ids"],
    "attention_mask": tokenized["attention_mask"],
    "trigger_indices_1": torch.tensor(event_triggers1),
    "trigger_indices_2": torch.tensor(event_triggers2),
    "labels": torch.tensor(labels)
    })

    return dataset

train_dataset = load_data("data 2/event_pairs.train")
dev_dataset = load_data("data 2/event_pairs.train")
test_dataset = load_data("data 2/event_pairs.train")

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2) # define number of labels (e.g., 2 for binary classification)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=["query", "value"] 
)

# configuring LORA to fine-tune only a small subset of parameters, drastically reducing memory usage and training time

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # get predicted class indices
    
    # compute accuracy, precision, recall, F1
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir="./roberta_finetuned",  # where to save model
    evaluation_strategy="epoch",  # evaluate at the end of each epoch
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16 = False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./roberta_finetuned_lora")

results = trainer.evaluate()
print(results)