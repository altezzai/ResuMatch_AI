import requests
import os

# Flask API Base URL (Update if hosted remotely)
BASE_URL = "http://127.0.0.1:5000"

def get_job_criteria(job_description):
    """
    Sends a job description to the Flask API and retrieves job criteria.
    """
    url = f"{BASE_URL}/get_job_criteria"
    data = {"job_description": job_description}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an error for non-200 status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "Failed to fetch job criteria", "details": str(e)}

def upload_cvs(cv_folder):
    """
    Uploads multiple CVs from a folder to the API.
    """
    url = f"{BASE_URL}/upload_cvs"
    files = []
    file_objects=[]

    if not os.path.exists(cv_folder):
        return {"error": f"Folder not found: {cv_folder}"}
    
    pdf_files = [f for f in os.listdir(cv_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        return {"error": "No PDF files found in the folder"}

    try:
        for file_name in pdf_files:
            file_path = os.path.join(cv_folder, file_name)
            file_obj = open(file_path, "rb")  # Open file
            file_objects.append(file_obj)  # Store in list
            files.append(("cv_files", (file_name, file_obj, "application/pdf")))

        response = requests.post(url, files=files)
        response.raise_for_status()  # Raise an error for non-200 status codes

        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "Failed to upload CVs", "details": str(e)}


    finally:
        # Close all opened files after the request is sent
        for file_obj in file_objects:
            file_obj.close()


if __name__ == "__main__":
    job_desc = "Looking for a civil engineer with 2 years experience"
    print("Fetching Job Criteria...")
    job_criteria = get_job_criteria(job_desc)
    print(job_criteria)

    cv_folder_path = r"D:\deepseek\sampleCVs"
    print("Uploading CVs for analysis...")
    upload_response = upload_cvs(cv_folder_path)
    print(upload_response)
