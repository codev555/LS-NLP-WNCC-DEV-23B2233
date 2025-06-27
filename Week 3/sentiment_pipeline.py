import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the IMDb dataset
dataset = load_dataset("imdb")

# 2. Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 3. Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

# 4. Preprocess dataset
encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset = encoded_dataset.remove_columns(["text"])
encoded_dataset.set_format("torch")

# 5. Evaluation metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, logging

# Optional: enable informative logging for progress display
logging.set_verbosity_info()

# 6. TrainingArguments (with logging and tqdm settings)
training_args = TrainingArguments(
    output_dir="./results",
    run_name="sentiment_analysis_run",     # avoids wandb warning
    eval_strategy="epoch",                 # for evaluation each epoch
    save_strategy="epoch",                 # save checkpoint each epoch
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,                      # log every 10 steps
    logging_first_step=True,               # also log at step 1
    report_to="none",                      # disable wandb/tensorboard
    disable_tqdm=False,                    # enable progress bar
)

# 7. Trainer setup (no tokenizer= to avoid deprecation warning)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=encoded_dataset["test"].select(range(500)),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8. Train the model
trainer.train()

# 9. Save model
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

# 10. Load and test on sample
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "positive" if prediction == 1 else "negative"

# Sample prediction
print("Sample prediction:", predict("The movie was absolutely amazing!"))
