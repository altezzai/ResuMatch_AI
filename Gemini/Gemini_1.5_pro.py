from flask import Flask,request, jsonify
import google.generativeai as genai
import json 

app=Flask(__name__)

genai.configure(api_key="AIzaSyCwvRvcYZoISHwue-0Ijlrw_GLSMm2lTuQ")

def run_gemini_model(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)  
        model_output = response.text.strip()
        print("Raw Model Output:", model_output)
          # Ensure JSON validity
        if model_output.startswith("```json"):
            model_output = model_output[7:-3].strip()  # Remove triple backticks and json tag

        try:
            return json.loads(model_output)
        except json.JSONDecodeError as e:
            return {"error": "Invalid JSON from Gemini", "details": str(e), "raw_output": model_output}
    except Exception as e:
        return {"error": "Gemini API execution failed", "details": str(e)}

def determine_priority(job_description):
    """
    Analyze job description and determine the most important section (Skills, Education, Experience).
    """
    keywords = {
        "skills": ["Python", "Machine Learning", "AI", "programming", "software", "development", "framework"],
        "education": ["PhD", "Master", "B.Sc", "degree", "qualification", "graduate"],
        "experience": ["years", "industry", "work", "projects", "professional"]
        
    }

    scores = {"Skills": 0, "Education": 0, "Experience": 0}

    for section, words in keywords.items():
        for word in words:
            if word.lower() in job_description.lower():
                scores[section.capitalize()] += 1

    # Sort sections based on their score in descending order
    sorted_sections = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [section[0] for section in sorted_sections]  # Return section names in order of importance

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Job Criteria API!'}), 200

@app.route('/get_job_criteria', methods=['POST'])
def get_job_criteria():
    # Get JSON data from the request
    data = request.get_json()
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description is missing'}), 400
    job_description = data['job_description']

    # Get priority order dynamically
    priority_order = determine_priority(job_description)

    prompt = f"""
    Based on the following job description, provide job selection criteria:
    "{job_description}"
    Return a valid JSON object with the following keys: "Skills", "Education", "Experience", "Certifications","Projects", "Other Relevant Criteria".
    Ensure that criteria are sorted based on importance: {priority_order}.
    Each key should contain a list of relevant single words or short phrases (e.g., "Python", "M.Sc Data Science", "Machine Learning").
    Do not include full sentences.
    """

    # Run the Gemini model with the prompt
    criteria = run_gemini_model(prompt)
    if "error" in criteria:
        return jsonify(criteria), 500

    # Reorder criteria based on priority
    sorted_criteria = {key: criteria.get(key, []) for key in priority_order}
    sorted_criteria.update({key: criteria.get(key, []) for key in criteria if key not in priority_order})
    return jsonify({'job_criteria': sorted_criteria})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)