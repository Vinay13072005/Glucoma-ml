import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# Use row 1 (index=1) as the header
os_df = pd.read_excel(r"C:\Users\VINAY REDDY\Downloads\PAPILA\PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f\ClinicalData\patient_data_os.xlsx", header=1)
od_df = pd.read_excel(r"C:\Users\VINAY REDDY\Downloads\PAPILA\PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f\ClinicalData\patient_data_od.xlsx", header=1)

# Drop any fully empty rows if present
os_df = os_df.dropna(how='all')
od_df = od_df.dropna(how='all')

# Clean column names
os_df.columns = [col.strip().lower().replace(" ", "_") for col in os_df.columns]
od_df.columns = [col.strip().lower().replace(" ", "_") for col in od_df.columns]

# Add source
os_df['source'] = 'OS'
od_df['source'] = 'OD'

# Combine
df = pd.concat([os_df, od_df], ignore_index=True)

def convert_row_to_text(row):
    return (
        f"Age: {row['age']}, Gender: {'Male' if row['gender'] == 0 else 'Female'}, "
        f"Dioptre 1: {row['dioptre_1']}, Dioptre 2: {row['dioptre_2']}, Astigmatism: {row.get('astigmatism', 'NA')}, "
        f"Phakic: {row['phakic/pseudophakic']}, IOP Pneumatic: {row['pneumatic']}, Perkins: {row['perkins']}, "
        f"Pachymetry: {row['pachymetry']}, Axial Length: {row['axial_length']}, VF_MD: {row['vf_md']}"
    )

# Drop rows with missing diagnosis
df = df[df['diagnosis'].notna()]

# Create new text column
df['text'] = df.apply(convert_row_to_text, axis=1)

# Final input dataframe
df_final = df[['text', 'diagnosis']]



# Step 1: Encode labels
label_encoder = LabelEncoder()
df_final['label'] = label_encoder.fit_transform(df_final['diagnosis'])

# Step 2: Train-test split
train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)

# Step 3: Tokenize using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = Dataset.from_pandas(train_df[['text', 'label']])
test_ds = Dataset.from_pandas(test_df[['text', 'label']])

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Step 4: Load BERT model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Step 5: Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer
)

# Step 6: Train the model
trainer.train()

# Step 7: Evaluate
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)


# Save model and tokenizer
model.save_pretrained("bert_glaucoma_model")
tokenizer.save_pretrained("bert_glaucoma_model")
