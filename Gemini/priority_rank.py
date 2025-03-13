import os
import pdfplumber
import docx
import re
import pandas as pd
from flashtext import KeywordProcessor

# Define weightage for each category
WEIGHTS = {
    "skills": 0.3,
    "experience": 0.3,
    "education": 0.2,
    "certifications": 0.1,
    "projects": 0.1,
    "job_gap_penalty": -0.05  # Negative weight for job gaps
}

# Predefined criteria for ranking
PREDEFINED_CRITERIA = {
    "skills": ["python", "machine learning", "deep learning", "data analysis", "nlp", "computer vision", "sql", "java", "c++", "cloud computing"],
    "experience": [],  # Experience is handled separately
    "education": ["bachelor", "master", "phd", "msc", "b.tech", "m.tech", "mba", "bca", "mca"],
    "certifications": ["aws", "azure", "gcp", "tensorflow", "pmp", "scrum", "cisco", "compTIA", "oracle certified"],
    "projects": ["data science", "nlp", "computer vision", "big data", "ai research", "robotics", "cybersecurity", "blockchain"],
    "languages": ["english", "spanish", "french", "german", "mandarin"],
    "tools": ["excel", "power bi", "tableau", "git", "docker", "kubernetes"]
}

def extract_text_from_resume(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + " "
    return text.lower().strip() if text else ""

def extract_keywords(text, keyword_list):
    if not text:
        return []  
    keyword_processor = KeywordProcessor()
    for keyword in keyword_list:
        keyword_processor.add_keyword(keyword.lower())
    return keyword_processor.extract_keywords(text)

def calculate_experience(text):
    years = re.findall(r'(\d+)\s*(years?|yrs?)', text)
    total_experience = sum(int(y[0]) for y in years) if years else 0
    return total_experience

def calculate_job_gaps(text):
    gaps = re.findall(r'gap of (\d+)\s*(months?|years?)', text)
    total_gaps = sum(int(g[0]) for g in gaps) if gaps else 0
    return total_gaps

def score_resume(resume_text, job_description, criteria):
    if not resume_text:
        return 0
    job_keywords = extract_keywords(job_description, criteria)
    resume_keywords = extract_keywords(resume_text, criteria)
    matching_keywords = set(resume_keywords) & set(job_keywords)
    return len(matching_keywords) / len(job_keywords) if job_keywords else 0

def rank_resumes(resume_folder, job_description, criteria=PREDEFINED_CRITERIA):
    scores = []
    for file in os.listdir(resume_folder):
        if file.endswith(".pdf") or file.endswith(".docx"):
            file_path = os.path.join(resume_folder, file)
            resume_text = extract_text_from_resume(file_path)
            if not resume_text:
                continue 
            
            skill_score = score_resume(resume_text, job_description, criteria['skills']) * WEIGHTS['skills']
            experience_score = calculate_experience(resume_text) * WEIGHTS['experience']
            education_score = score_resume(resume_text, job_description, criteria['education']) * WEIGHTS['education']
            certification_score = score_resume(resume_text, job_description, criteria['certifications']) * WEIGHTS['certifications']
            project_score = score_resume(resume_text, job_description, criteria['projects']) * WEIGHTS['projects']
            job_gap_penalty = calculate_job_gaps(resume_text) * WEIGHTS['job_gap_penalty']
            
            total_score = (skill_score + experience_score + education_score +
                           certification_score + project_score + job_gap_penalty)

            scores.append({
                "resume": file,
                "total_score": round(total_score, 2),
                "skills_score": round(skill_score, 2),
                "experience_score": round(experience_score, 2),
                "education_score": round(education_score, 2),
                "certifications_score": round(certification_score, 2),
                "projects_score": round(project_score, 2),
                "job_gap_penalty": round(job_gap_penalty, 2)
            })
    
    ranked_resumes = sorted(scores, key=lambda x: x["total_score"], reverse=True)
    return ranked_resumes

# Example Usage
resume_folder = r"D:\deepseek\sampleCVs"   # Update with actual path
job_description = "Looking for a python developer. MCA is prefered. should have 2 years experience"

ranked_resumes = rank_resumes(resume_folder, job_description)

# Convert to DataFrame for better readability
df = pd.DataFrame(ranked_resumes)
print(df)
