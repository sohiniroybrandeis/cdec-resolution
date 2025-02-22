import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from torch.utils.data import Dataset
from datasets import Dataset
from transformers import TrainingArguments, Trainer, LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import gc

gc.collect()

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

def align_token_indices(sentence, tokenizer, original_index):
    """
    Maps original word index to tokenized index.
    """
    words = sentence.split()
    tokens = tokenizer.tokenize(sentence)
    
    #to char index
    char_count = 0
    word_to_char_idx = []
    for word in words:
        word_to_char_idx.append(char_count)
        char_count += len(word) + 1  # +1 for space

    # character span of the target word
    if original_index == -1:
        return -1 #missing
    start_char_idx = word_to_char_idx[original_index]

    # to token index
    token_char_count = 0
    for i, token in enumerate(tokens):
        if token.startswith("▁"):
            token_char_count += len(token) - 1
        else:
            token_char_count += len(token)

        if token_char_count >= start_char_idx:
            return i 

    return -1


with open("data 2/event_pairs.train", "r", encoding="utf-8") as file:
    for line in file:
        tokens = line.split('\t')

        aligned_trigger1 = align_token_indices(tokens[0], tokenizer, int(tokens[1]))
        aligned_trigger2 = align_token_indices(tokens[11], tokenizer, int(tokens[12]))

        train_sentences1.append(tokens[0])
        train_sentences2.append(tokens[11])
        train_event_triggers1.append(aligned_trigger1)
        train_event_triggers2.append(aligned_trigger2)

        train_labels.append(int(tokens[22])) #deal with the other info later

# print(train_sentences)
train_texts = [s1 + " " + s2 for s1, s2 in zip(train_sentences1, train_sentences2)]
tr_tokenized = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")


with open("data 2/event_pairs.dev", "r", encoding="utf-8") as file:
    for line in file:
        tokens = line.split('\t')

        aligned_trigger1 = align_token_indices(tokens[0], tokenizer, int(tokens[1]))
        aligned_trigger2 = align_token_indices(tokens[11], tokenizer, int(tokens[12]))

        dev_sentences1.append(tokens[0])
        dev_sentences2.append(tokens[11])

        dev_event_triggers1.append(aligned_trigger1)
        dev_event_triggers2.append(aligned_trigger2)

        dev_labels.append(int(tokens[22])) #deal with the other info later

dev_texts = [s1 + " " + s2 for s1, s2 in zip(dev_sentences1, dev_sentences2)]
dev_tokenized = tokenizer(dev_texts, padding=True, truncation=True, return_tensors="pt")

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
model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", num_labels=2)
model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Change to TaskType.TOKEN_CLS for token classification
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get predicted class indices
    
    # Compute accuracy, precision, recall, F1
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir="./llama_finetuned",  # Where to save model
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Your training data
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./llama_finetuned_lora")

# trainer.evaluate()

results = trainer.evaluate()
print(results)