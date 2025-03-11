import torch
import numpy as np
import gc
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") #create tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #account for PAD token


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
            triggers1.append(int(tokens[1]))
            triggers2.append(int(tokens[12]))
            labels.append(int(tokens[22]))

    texts = [s1 + " " + s2 for s1, s2 in zip(sentences1, sentences2)] #concatenate sentences
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "trigger_indices_1": torch.tensor(triggers1),
        "trigger_indices_2": torch.tensor(triggers2),
        "labels": torch.tensor(labels)
    })

    return dataset #huggingface dict

#load data from train and dev
train_dataset = load_data("data2/event_pairs.train")
dev_dataset = load_data("data2/event_pairs.dev")
test_dataset = load_data("data2/event_pairs.train")

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="bfloat16"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-1B",
#     num_labels=2,
#     device_map="auto",  # Automatically distribute across available GPUs
#     quantization_config=quantization_config
# )

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    num_labels=2,
    device_map="auto"  # Automatically distribute across available GPUs
)
# quantization_config=quantization_config

model.resize_token_embeddings(len(tokenizer))

# Apply LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=2,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

training_args = SFTConfig(
    output_dir = "./llama_finetuned",
    max_seq_length=256,
    packing=True,
    bf16=True,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_gpu_eval_batch_size=8,
    do_eval=True,
    eval_strategy="epoch",
    save_total_limit=2  # Avoid excessive checkpoint saving
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("./llama_finetuned_lora")




# Evaluation (run separately)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    num_labels=2,
    device_map="auto"  # Automatically distribute across available GPUs
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=2,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

def generate_prompt(example):
    # get features from your custom dataset to build the prompt
    mention1 = example["mention1"]
    mention2 = example["mention2"]
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    prompt = (
    f"Question: Is event {mention1} in sentence1 and event {mention2} in sentence2 refer to the same event?\n"
    f"Sentence1: {sentence1}\n"
    f"Sentence2: {sentence2}\n"
    f"Answer:"
    )
    return prompt

def parse_predicted_label(generated_text):
# Based on your prompt and custom dataset, parse the response
# Simple approach: everything after "Answer:"
    if "Answer:" in generated_text:
        answer_part = generated_text.split("Answer:")[-1].strip()
    if answer_part.startswith("0"):
        return 0
    else:
        return 1
    return 0 # fallback if the structure is unexpected

def evaluate(test_set, model, tokenizer):
    pred_labels = []
    gold_labels = []
    for example in test_set:
        gold_label = example["label"] # get the gold label from your custom dataset
        gold_labels.append(gold_label)
        prompt = generate_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
    # set max_new_tokens low to avoid OOM
        outputs = model.generate(**inputs, max_new_tokens=5)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_label = parse_predicted_label(generated_text)
        pred_labels.append(pred_label)
    # now calculate the metrics
    result = compute_metrics(gold_labels, pred_labels)
    return result
