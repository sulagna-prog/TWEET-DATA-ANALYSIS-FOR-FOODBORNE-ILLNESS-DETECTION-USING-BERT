import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pickle

df = pd.read_csv('Restaurant.csv')
texts = df['text'].tolist()
labels = df['sentiment'].tolist()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
dataset = torch.utils.data.TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], torch.tensor(labels))
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(5):
    total_loss = 0

    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader)}")

model_path = 'bert_sentiment_model.pkl'
tokenizer_path = 'bert_tokenizer.pkl'

torch.save(model, model_path)
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
