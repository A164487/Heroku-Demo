import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={ 'status': 2, 'edu': 70 , 'occ': 2, 'twocars': 0, 'threecars': 1,'ac': 1, 'laptop': 1, 'hp': 1, 'tv': 1, 
							 'internet': 1, 'water': 1, 'ownlq': 1, 'ownership': 2, 'person': 6, 'household': 19876, 'income':8554 })

print(r.json())