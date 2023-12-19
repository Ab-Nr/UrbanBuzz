import requests
from bs4 import BeautifulSoup

def scrape_tourist_spots(urls):
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            tourist_spots = soup.find_all('div', class_='tourist-spot')
            for spot in tourist_spots:
                print(f'Tourist Spot from {url}: {spot.text.strip()}')
        else:
            print(f'Failed to fetch tourist spots from {url}: {response.status_code}')

def scrape_fashion(urls):
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            fashion_items = soup.find_all('div', class_='fashion-item')
            for item in fashion_items:
                print(f'Fashion Item from {url}: {item.text.strip()}')
        else:
            print(f'Failed to fetch fashion items from {url}: {response.status_code}')

def scrape_food_and_restaurants(urls):
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            restaurants = soup.find_all('div', class_='restaurant')
            for restaurant in restaurants:
                print(f'Restaurant from {url}: {restaurant.text.strip()}')
        else:
            print(f'Failed to fetch restaurants from {url}: {response.status_code}')

tourist_spots_urls = ['https://www.facebook.com/best.traveldestinations/', 'https://twitter.com/T3_UBM']
fashion_urls = ['https://twitter.com/fastytrends?lang=en', 'https://www.facebook.com/TrendingFashion15/']
food_and_restaurants_urls = ['https://www.facebook.com/p/Street-Food-Around-The-World-100072389212192/','https://twitter.com/TheWorlds50Best','https://www.yelp.de/search?find_desc=Restaurants&find_loc=New+York%2C+NY%2C+Vereinigte+Staaten']

scrape_tourist_spots(tourist_spots_urls)
scrape_fashion(fashion_urls)
scrape_food_and_restaurants(food_and_restaurants_urls)
