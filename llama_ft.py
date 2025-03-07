import torch
import numpy as np
import gc
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def align_token_indices(sentence, tokenizer, original_index):
    words = sentence.split()
    tokens = tokenizer.tokenize(sentence)
    
    char_count = 0
    word_to_char_idx = [char_count]
    for word in words:
        char_count += len(word) + 1  # +1 for space
        word_to_char_idx.append(char_count)

    if original_index == -1:
        return -1  # Missing index
    
    start_char_idx = word_to_char_idx[original_index]

    token_char_count = 0
    for i, token in enumerate(tokens):
        token_char_count += len(token) - 1 if token.startswith("▁") else len(token)
        if token_char_count >= start_char_idx:
            return i 

    return -1


def load_data(file_path):
    sentences1 = []
    sentences2 = []
    triggers1 = []
    triggers2 = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            tokens = line.split('\t')
            sentences1.append(tokens[0])
            sentences2.append(tokens[11])
            triggers1.append(align_token_indices(tokens[0], tokenizer, int(tokens[1])))
            triggers2.append(align_token_indices(tokens[11], tokenizer, int(tokens[12])))
            labels.append(int(tokens[22]))

    texts = [s1 + " " + s2 for s1, s2 in zip(sentences1, sentences2)] #concatenate sentences
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "trigger_indices_1": torch.tensor(triggers1),
        "trigger_indices_2": torch.tensor(triggers2),
        "labels": torch.tensor(labels)
    })

    return dataset

#load data from train and dev
train_dataset = load_data("data2/event_pairs.train")
dev_dataset = load_data("data2/event_pairs.dev")
test_dataset = load_data("data2/event_pairs.train")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    num_labels=2,
    device_map="auto",  # Automatically distribute across available GPUs
    quantization_config=quantization_config
)

model.resize_token_embeddings(len(tokenizer))

# Apply LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.to(device)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,  # Reduce batch size for lower memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    dataloader_num_workers=0,
    fp16=True,  # Enable mixed precision
    save_total_limit=2  # Avoid excessive checkpoint saving
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("./llama_finetuned_lora")

# Evaluation
results = trainer.evaluate()
print(results)
