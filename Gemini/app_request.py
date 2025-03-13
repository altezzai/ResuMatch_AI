import requests
import os

# Flask API Base URL (update if hosted remotely)
BASE_URL = "http://127.0.0.1:5000"  

def get_job_criteria(job_description):
    """
    Sends a job description to the Flask API and retrieves job criteria.
    """
    url = f"{BASE_URL}/get_job_criteria"
    data = {"job_description": job_description}
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch job criteria, Status Code: {response.status_code}", "details": response.text}

def upload_cvs(cv_folder):
    """
    Uploads multiple CVs from a folder to the API.
    """
    url = f"{BASE_URL}/upload_cvs"
    files = []
    
    for file_name in os.listdir(cv_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(cv_folder, file_name)
            files.append(("cv_files", (file_name, open(file_path, "rb"), "application/pdf")))

    response = requests.post(url, files=files)

    # Close opened files
    for _, file_obj in files:
        file_obj[1].close()

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to upload CVs, Status Code: {response.status_code}", "details": response.text}

if __name__ == "__main__":
    job_desc = "Looking for a Python Developer with expertise in Machine Learning, SQL, and Data Analysis."
    print("Fetching Job Criteria...")
    job_criteria = get_job_criteria(job_desc)
    print(job_criteria)

    cv_folder_path =r"D:\deepseek\sampleCVs"  # Update with actual path
    print("Uploading CVs for analysis...")
    upload_response = upload_cvs(cv_folder_path)
    print(upload_response)
