import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AdamW
from tqdm import tqdm

gps_data = {'latitude': 37.7749, 'longitude': -122.4194}

scraped_data = [
    {'text': 'Positive event', 'label': 'positive'},
    {'text': 'Negative event', 'label': 'negative'},
    {'text': 'Neutral event', 'label': 'neutral'},
]

import pandas as pd
df = pd.DataFrame(scraped_data)

ml_data = [
    {'text': 'Another positive event'},
    {'text': 'Another negative event'},
    {'text': 'Another neutral event'},
]

def get_gps_data():
    return {'latitude': 37.7749, 'longitude': -122.4194}

def scrape_data_near_location(latitude, longitude, category):

    category_urls = {
        'events': f'https://example.com/events?lat={latitude}&long={longitude}',
        'fashion': f'https://example.com/fashion?lat={latitude}&long={longitude}',
        'food': f'https://example.com/food?lat={latitude}&long={longitude}',
        'tourist_spots': f'https://example.com/tourist_spots?lat={latitude}&long={longitude}',
    }

    url = category_urls.get(category, '')
    if not url:
        raise ValueError(f'Invalid category: {category}')

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    if category == 'events':
        data = [{'text': event.text} for event in soup.find_all('div', class_='event-text')]
    elif category == 'fashion':
        data = [{'text': fashion.text} for fashion in soup.find_all('div', class_='fashion-text')]
    elif category == 'food':
        data = [{'text': restaurant.text} for restaurant in soup.find_all('div', class_='restaurant-text')]
    elif category == 'tourist_spots':
        data = [{'text': spot.text} for spot in soup.find_all('div', class_='spot-text')]

    return data

def perform_sentiment_analysis(texts):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3) 

    encoded_data = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()


    new_encoded_data = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    new_dataset = SentimentDataset(new_encoded_data, torch.zeros(len(texts)))  
    new_loader = DataLoader(new_dataset, batch_size=8, shuffle=False)

    new_predictions = []
    with torch.no_grad():
        for batch in new_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            new_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

    return new_predictions

def main():
    # Get GPS data
    user_location = get_gps_data()

    #Scrape data near the user's location for different categories
    categories = ['events', 'fashion', 'food', 'tourist_spots']
    all_category_data = {}
    for category in categories:
        category_data = scrape_data_near_location(user_location['latitude'], user_location['longitude'], category)
        all_category_data[category] = category_data

    #Extract text for sentiment analysis
    all_texts = {category: [item['text'] for item in data] for category, data in all_category_data.items()}

    #Perform sentiment analysis on the texts for each category
    all_sentiment_predictions = {category: perform_sentiment_analysis(texts) for category, texts in all_texts.items()}
