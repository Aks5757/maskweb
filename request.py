import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json=dict(petal_length=2, petal_width=9, sepal_length=6, sepal_width=6))

print(r.json())