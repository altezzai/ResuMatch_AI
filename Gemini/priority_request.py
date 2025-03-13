import requests
import json

# API Endpoint
url = "http://127.0.0.1:5000/get_job_criteria"

# Sample Job Description
job_description = """
We are looking for a senior accountant.
"""

# Request payload
data = {"job_description": job_description}

# Send POST request
response = requests.post(url, json=data)

# Print response
print("Response Status Code:", response.status_code)
print(json.dumps(response.json(), indent=4))
