import requests

url = 'http://localhost:5000/api/chat'  # Updated endpoint
data = {'question': 'excessive urination and excessive thrust and hunger, from last 4 months, i have also experienced numbeness or a burning sensation in my extremities, yes blood sugar, no recent injuries'}
response = requests.post(url, json=data)

print(response.json())