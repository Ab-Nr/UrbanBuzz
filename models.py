import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

scraped_data = [
    {'text': 'Positive event', 'label': 'positive'},
    {'text': 'Negative event', 'label': 'negative'},
    {'text': 'Neutral event', 'label': 'neutral'},
]

import pandas as pd
df = pd.DataFrame(scraped_data)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3) 

encoded_data = tokenizer(df['text'].tolist(), truncation=True, padding=True, return_tensors='pt')

label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
labels = df['label'].map(label_mapping)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings, test_encodings, train_labels, test_labels = train_test_split(encoded_data, labels, test_size=0.2, random_state=42)

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

num_epochs = 3 

for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')

model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy: {accuracy}')

new_scraped_data = [
    {'text': 'Another positive event'},
    {'text': 'Another negative event'},
    {'text': 'Another neutral event'},
]

new_encoded_data = tokenizer([item['text'] for item in new_scraped_data], truncation=True, padding=True, return_tensors='pt')

new_dataset = SentimentDataset(new_encoded_data, torch.zeros(len(new_scraped_data)))  
new_loader = DataLoader(new_dataset, batch_size=8, shuffle=False)

new_predictions = []
with torch.no_grad():
    for batch in new_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        new_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

print('Predictions for new data:')
for text, prediction in zip([item['text'] for item in new_scraped_data], new_predictions):
    print(f'Text: {text}, Prediction: {prediction}')
