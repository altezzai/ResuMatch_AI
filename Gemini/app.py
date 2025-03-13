from flask import Flask, request, jsonify
import google.generativeai as genai
import json
import os
import pdfplumber
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyBPbbjGi09zziCO0i5YzgrAlQshE9D1zmM")  # Replace with actual API key

def run_gemini_model(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        model_output = response.text.strip()

        if model_output.startswith("```json"):
            model_output = model_output[7:-3].strip()
        
        try:
            return json.loads(model_output)
        except json.JSONDecodeError as e:
            return {"error": "Invalid JSON from Gemini", "details": str(e), "raw_output": model_output}
    except Exception as e:
        return {"error": "Gemini API execution failed", "details": str(e)}

def determine_priority(job_description):
    keywords = {
        "Skills": ["Python", "Machine Learning", "AI", "programming", "software", "development", "framework"],
        "Education": ["PhD", "Master", "B.Sc", "degree", "qualification", "graduate"],
        "Experience": ["years", "industry", "work", "projects", "professional"],
        "Projects": ["portfolio", "open-source", "Kaggle", "GitHub", "research", "competition"],
        "Certifications": ["certified", "certification", "AWS", "Google Cloud", "Azure"],
        "Other Relevant Criteria": ["problem-solving", "teamwork", "communication", "leadership"]
    }
    
    scores = {key: sum(job_description.lower().count(word.lower()) for word in words) for key, words in keywords.items()}
    
    sorted_sections = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {category: index + 1 for index, (category, _) in enumerate(sorted_sections)}

@app.route('/get_job_criteria', methods=['POST'])
def get_job_criteria():
    data = request.get_json()
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description is missing'}), 400
    job_description = data['job_description']
    priority_mapping = determine_priority(job_description)

    prompt = f"""
    Based on the following job description, provide job selection criteria:
    "{job_description}"
    Return a valid JSON object with the following keys: "Skills", "Education", "Experience", "Certifications", "Projects", "Other Relevant Criteria".
    """
    
    criteria = run_gemini_model(prompt)
    if "error" in criteria:
        return jsonify(criteria), 500
    
    final_criteria = {key: criteria.get(key, []) for key in priority_mapping.keys()}
    prioritized_criteria = {key: {"priority": priority_mapping[key], "items": value} for key, value in final_criteria.items()}
    return jsonify({'job_criteria': prioritized_criteria}), 200

# Load spaCy NLP Model
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

KEYWORDS = {
    "skills": ["python", "sql", "java", "machine learning", "deep learning", "data analysis"],
    "education": ["bachelor", "master", "phd", "degree", "b.sc", "m.sc", "mca", "b.tech", "m.tech"],
    "certification": ["certified", "certification", "aws", "azure", "pmp", "csm"],
    "projects": ["project", "developed", "built", "created", "implemented"],
    "experience": ["year", "experience", "worked", "internship"],
    "job_gap": ["gap", "break", "sabbatical"]
}

for label, words in KEYWORDS.items():
    matcher.add(label.upper(), [nlp(word) for word in words])

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.lower()

def extract_keywords(text):
    doc = nlp(text)
    matches = matcher(doc)
    extracted = {key: [] for key in KEYWORDS}
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id].lower()
        extracted[label].append(doc[start:end].text.lower())
    return {key: list(set(value)) for key, value in extracted.items()}

def score_resume(extracted_data, job_keywords):
    score_details = {
        "Skill Score": len(set(extracted_data["skills"]) & set(job_keywords)) * 5,
        "Education Score": len(extracted_data["education"]) * 10,
        "Experience Score": sum([int(word) for word in extracted_data["experience"] if word.isdigit()]) * 3,
        "Certifications Score": len(extracted_data["certification"]) * 5,
        "Projects Score": len(extracted_data["projects"]) * 3,
        "Job Gap Penalty": -len(extracted_data["job_gap"]) * 5,
    }
    total_score = sum(score_details.values())
    return total_score, score_details


@app.route('/upload_cvs', methods=['POST'])
def upload_cvs():
    if 'cv_files' not in request.files:
        return jsonify({'error': 'No CV files uploaded'}), 400
    
    files = request.files.getlist('cv_files')
    
    if not files:
        return jsonify({'error': 'No CV files received'}), 400

    uploaded_files = [file.filename for file in files]
    
    # Ensure directory exists
    upload_folder = "uploaded_cvs"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Example job keywords (should be dynamically retrieved)
    job_keywords = ["python", "machine learning", "data analysis", "sql"]
    
    ranked_candidates = []

    for file in files:
        if file.filename.endswith(".pdf"):
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            text = extract_text_from_pdf(file_path)
            extracted_data = extract_keywords(text)
            total_score, score_details = score_resume(extracted_data, job_keywords)

            ranked_candidates.append({
                "Name": file.filename,
                **score_details,
                "Total Score": round(total_score, 2)
            })

    # Sort ranked candidates based on score
    ranked_candidates = sorted(ranked_candidates, key=lambda x: x["Total Score"], reverse=True)

    return jsonify({
        "message": "CVs uploaded successfully",
        "files": uploaded_files,
        "ranked_candidates": ranked_candidates})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

