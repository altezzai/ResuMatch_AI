import requests
import json

API_URL = "http://127.0.0.1:5000/get_job_criteria"

job_description = """
We are hiring a senior accountant . 
"""

# Make POST request to the API
response = requests.post(API_URL, json={"job_description": job_description})

# Print the response
if response.status_code == 200:
    print(json.dumps(response.json(), indent=4))
else:
    print("Error:", response.json())
