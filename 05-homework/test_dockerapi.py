import requests

url = "http://127.0.0.1:80/predict/"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
answer = requests.post(url, json=client).json()
print(answer)