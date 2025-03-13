from flask import Flask, request, jsonify
import google.generativeai as genai
import json 

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyBPbbjGi09zziCO0i5YzgrAlQshE9D1zmM")  # Replace with actual API key

def run_gemini_model(prompt):
    """
    Calls the Gemini model to generate job criteria based on the job description.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)  
        model_output = response.text.strip()
        
        # Ensure JSON validity
        if model_output.startswith("```json"):
            model_output = model_output[7:-3].strip()  # Remove triple backticks and JSON tag
        
        try:
            return json.loads(model_output)
        except json.JSONDecodeError as e:
            return {"error": "Invalid JSON from Gemini", "details": str(e), "raw_output": model_output}
    except Exception as e:
        return {"error": "Gemini API execution failed", "details": str(e)}

def determine_priority(job_description):
    """
    Dynamically determines priority based on keyword occurrence in the job description.
    """
    keywords = {
        "Skills": ["Python", "Machine Learning", "AI", "programming", "software", "development", "framework"],
        "Education": ["PhD", "Master", "B.Sc", "degree", "qualification", "graduate"],
        "Experience": ["years", "industry", "work", "projects", "professional"],
        "Projects": ["portfolio", "open-source", "Kaggle", "GitHub", "research", "competition"],
        "Certifications": ["certified", "certification", "AWS", "Google Cloud", "Azure", "professional certification"],
        "Other Relevant Criteria": ["problem-solving", "teamwork", "communication", "leadership", "critical thinking"]
    }

    scores = {key: 0 for key in keywords}  # Initialize scores

    for category, words in keywords.items():
        for word in words:
            scores[category] += job_description.lower().count(word.lower())

    # Sort by descending importance
    sorted_sections = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Assign priority numbers dynamically
    priority_mapping = {category: index + 1 for index, (category, _) in enumerate(sorted_sections)}
    
    return priority_mapping

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Job Criteria API!'}), 200

@app.route('/get_job_criteria', methods=['POST'])
def get_job_criteria():
    """
    Extracts job selection criteria from a job description using Gemini AI.
    """
    data = request.get_json()
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description is missing'}), 400
    job_description = data['job_description']

    # Determine dynamic priority based on job description content
    priority_mapping = determine_priority(job_description)

    prompt = f"""
    Based on the following job description, provide job selection criteria:
    "{job_description}"
    Return a valid JSON object with the following keys: "Skills", "Education", "Experience", "Certifications", "Projects", "Other Relevant Criteria".
    Each key should contain a list of relevant single words or short phrases (e.g., "Python", "M.Sc Data Science", "Machine Learning").
    """

    criteria = run_gemini_model(prompt)
    if "error" in criteria:
        return jsonify(criteria), 500

    # Ensure all required keys exist, filling missing ones with empty lists
    final_criteria = {key: criteria.get(key, []) for key in priority_mapping.keys()}

    # Apply dynamic priority
    prioritized_criteria = {
        key: {"priority": priority_mapping[key], "items": value}
        for key, value in final_criteria.items()
    }

    return jsonify({'job_criteria': prioritized_criteria}), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
