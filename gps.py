import requests
from geopy.geocoders import Nominatim

def get_nearby_places(api_key, latitude, longitude, category):
    endpoint = "https://api.yelp.com/v3/businesses/search"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "categories": category,
        "radius": 1000,  
        "limit": 5  
    }

    response = requests.get(endpoint, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        return data.get("businesses", [])
    else:
        print(f"Failed to fetch nearby places: {response.status_code}")
        return []

def main():
    yelp_api_key = "yelp_api_key"

    user_latitude = 40.7128
    user_longitude = -74.0060

    geolocator = Nominatim(user_agent="geofencing_app")
    location = geolocator.reverse((user_latitude, user_longitude), language='en')

    if location and 'address' in location.raw and 'city' in location.raw['address']:
        user_city = location.raw['address']['city']
        print(f"User is in {user_city}")

        restaurants = get_nearby_places(yelp_api_key, user_latitude, user_longitude, "restaurants")
        print("Nearby Restaurants:")
        for restaurant in restaurants:
            print(f"{restaurant['name']} - {restaurant['distance']} meters away")

        tourist_spots = get_nearby_places(yelp_api_key, user_latitude, user_longitude, "landmarks")
        print("Nearby Tourist Spots:")
        for tourist_spot in tourist_spots:
            print(f"{tourist_spot['name']} - {tourist_spot['distance']} meters away")
    else:
        print("City not found for the given coordinates.")

if __name__ == "__main__":
    main()
