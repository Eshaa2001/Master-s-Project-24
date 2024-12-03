import json

from datasets import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
)

# Loading the curated dataset
with open("fine_tuning_d3.json", "r") as f:
    file_content = f.read()
    data = json.loads(file_content)


# Converting data into a list of dictionaries
train_data = [
    {
        "input_text": f"Context: {item['context']} Query: {item['query']}",
        # "target_text": item["answer"],
        "target_text": f"Answer: {item['answer']} Source: {item['source']}",
    }
    for item in data
]

# Creating a Hugging Face Dataset
dataset = Dataset.from_list(train_data)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


# Tokenizing the dataset
def tokenize(batch):
    inputs = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
    )
    labels = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=300,
    )
    inputs["labels"] = labels["input_ids"]
    return inputs


tokenized_data = dataset.map(tokenize, batched=True)
train_test_split = tokenized_data.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model3",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=10,
    num_train_epochs=30,
    weight_decay=0.01,
    save_strategy="no",
)

# Initializing Trainer
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)"""

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Adding evaluation dataset
)

# Fine-tuning the model
trainer.train()

# Save the final model and tokenizer for later use
final_model_path = "./fine_tuned_model3/final_checkpoint"
trainer.save_model(final_model_path)  # Save model
tokenizer.save_pretrained(final_model_path)  # Save tokenizer
