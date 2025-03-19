import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text_to_fp
from io import StringIO
import docx
import spacy  # For NER
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_lg")  # Using a larger model for better NER
except ImportError:
    print("Downloading en_core_web_lg model for spaCy...")
    import spacy.cli
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Configure Gemini API
genai.configure(api_key="AIzaSyBPbbjGi09zziCO0i5YzgrAlQshE9D1zmM")  # Replace with your actual Gemini API key
model = genai.GenerativeModel('gemini-1.5-pro')

# --- Helper Functions ---
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    return ""

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            output_string = StringIO()
            extract_text_to_fp(pdf_file, output_string)
            return output_string.getvalue()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
        return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

def calculate_job_gap(resume_text):
    """Calculates job gap in years."""
    dates = re.findall(r'(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s*(\d{4})', resume_text)
    if not dates:
        return 0

    parsed_dates = []
    for match in dates:
        month_str = match[0]  # Full month string
        year_str = match[-1] # Year string
        try:
            date_obj = datetime.strptime(f"{month_str} {year_str}", "%B %Y").date()
            parsed_dates.append(date_obj)
        except ValueError:
            print(f"Warning: Could not parse date: {month_str} {year_str}")
            continue

    sorted_dates = sorted(parsed_dates)

    gaps = []
    for i in range(len(sorted_dates) - 1):
        gap = (sorted_dates[i+1] - sorted_dates[i]).days
        if gap > 90: # Consider a gap if it's more than 3 months
            gaps.append(gap / 365.25) # Convert days to years

    return sum(gaps)

def generate_job_analysis(job_description):
    prompt = f"""
    Analyze the following job description and provide the key criteria for evaluating candidates (skills, experience, education, certifications, projects, etc.) in JSON format. For each criterion, also list specific keywords or phrases associated with it.

    Job Description:
    {job_description}
    """
    try:
        response = model.generate_content(prompt)
        if response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                print("Error decoding job analysis JSON. Response was:", response.text)
                return {}
        else:
            print("Gemini API returned an empty response for job analysis.")
            return {}
    except Exception as e:
        print(f"Error during Gemini API call for job analysis: {e}")
        return {}

def calculate_resume_criteria_scores(resume_text, job_criteria):
    scores = {}
    resume_text_lower = resume_text.lower()
    resume_entities = extract_entities(resume_text)

    for criterion, keywords in job_criteria.items():
        scores[criterion] = 0
        if keywords:
            for keyword in keywords:
                if keyword.lower() in resume_text_lower:
                    scores[criterion] += 1
        # Add logic to score based on entities if applicable
        if criterion == 'skills':
            resume_skills = [skill.lower() for sublist in resume_entities.values() for skill in sublist]
            common_skills = set(resume_skills) & set([kw.lower() for kw in keywords])
            scores[criterion] += len(common_skills) * 2 # Increase score if entity match

    return scores

def train_word2vec_model(resumes_folder):
    sentences = []
    for filename in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, filename)
        text = extract_text(file_path)
        if text:
            sentences.append(nltk.word_tokenize(preprocess_text(text)))
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_resume_embedding(resume_text, word2vec_model):
    words = nltk.word_tokenize(preprocess_text(resume_text))
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def prepare_ml_data(resumes_folder, job_description, word2vec_model, labels):
    features = []
    processed_labels = []
    file_list = os.listdir(resumes_folder)
    num_files = len(file_list)
    num_labels = len(labels)

    if num_files > num_labels:
        print(f"Warning: More resumes ({num_files}) than labels ({num_labels}). Using the first {num_labels} labels.")
        file_list = file_list[:num_labels]
    elif num_labels > num_files:
        print(f"Warning: More labels ({num_labels}) than resumes ({num_files}). Using the first {num_files} labels.")
        labels = labels[:num_files]

    for i, filename in enumerate(file_list):
        file_path = os.path.join(resumes_folder, filename)
        resume_text = extract_text(file_path)
        if resume_text:
            resume_embedding = get_resume_embedding(resume_text, word2vec_model)
            job_embedding = get_resume_embedding(job_description, word2vec_model)
            similarity_embedding = cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]
            job_gap = calculate_job_gap(resume_text)
            features.append(np.concatenate([resume_embedding, job_embedding, [similarity_embedding, job_gap]]))
            processed_labels.append(labels[i])
    return features, processed_labels

def train_ranking_model_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(probability=True, kernel='linear') # Using a linear kernel for simplicity
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ranking model (SVM) accuracy: {accuracy:.4f}")
    return model, scaler

def predict_rankings_ml_svm(resumes_folder, job_description, word2vec_model, model, scaler, job_criteria):
    predictions = {}
    for filename in os.listdir(resumes_folder):
        file_path = os.path.join(resumes_folder, filename)
        resume_text = extract_text(file_path)
        if resume_text:
            resume_embedding = get_resume_embedding(resume_text, word2vec_model)
            job_embedding = get_resume_embedding(job_description, word2vec_model)
            similarity_embedding = cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]
            job_gap = calculate_job_gap(resume_text)
            feature_vector = np.concatenate([resume_embedding, job_embedding, [similarity_embedding, job_gap]]).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            prediction = model.predict_proba(feature_vector_scaled)[0][1] # Probability of being a good fit
            criteria_scores = calculate_resume_criteria_scores(resume_text, job_criteria)
            predictions[filename] = {"overall_score": prediction, "criteria_scores": criteria_scores}
    return predictions

if __name__ == "__main__":
    job_description = """
    We are looking for a skilled Python Developer with experience in machine learning and data analysis.
    The ideal candidate should have a Bachelor's degree in Computer Science or a related field.
    Experience with libraries like NumPy, Pandas, and Scikit-learn is highly desirable.
    Familiarity with cloud platforms (AWS, Azure, GCP) is a plus.
    Projects showcasing problem-solving skills and coding abilities are valued.
    """
    resumes_folder = r"D:\deepseek\sampleCVs"
    if not os.path.exists(resumes_folder):
        os.makedirs(resumes_folder)
        # Create dummy files for testing

    job_criteria_json = generate_job_analysis(job_description)
    print("Job Criteria (JSON):\n", json.dumps(job_criteria_json, indent=4))

    # --- Machine Learning Approach (SVM) ---
    try:
        word2vec_model = train_word2vec_model(resumes_folder)
        # Replace with your actual labels (1 for good fit, 0 for not)
        labels = [1, 0, 1, 0] * (len(os.listdir(resumes_folder)) // 4 or 1)
        features, labels = prepare_ml_data(resumes_folder, job_description, word2vec_model, labels)
        if features:
            if len(features) != len(labels):
                print(f"Warning: Number of features ({len(features)}) does not match number of labels ({len(labels)}).")
                min_len = min(len(features), len(labels))
                features = features[:min_len]
                labels = labels[:min_len]

            ranking_model_svm, scaler_svm = train_ranking_model_svm(features, labels)
            predictions_ml_svm = predict_rankings_ml_svm(resumes_folder, job_description, word2vec_model, ranking_model_svm, scaler_svm, job_criteria_json)
            ranked_resumes_ml_svm = sorted(predictions_ml_svm.items(), key=lambda item: item[1]["overall_score"], reverse=True)

            print("\nRanked Resumes (Machine Learning - SVM) with Detailed Scores:")
            print("| Rank | Filename | Overall Score |", end=" ")
            # Print the criteria as headers
            if ranked_resumes_ml_svm:
                first_resume_scores = ranked_resumes_ml_svm[0][1]['criteria_scores']
                for criterion in first_resume_scores:
                    print(f"{criterion.capitalize()} Score |", end=" ")
            print()
            print("|------|----------|---------------|", end=" ")
            if ranked_resumes_ml_svm:
                for _ in first_resume_scores:
                    print("--------------|", end=" ")
            print()

            rank = 1
            for filename, scores_dict in ranked_resumes_ml_svm:
                print(f"| {rank:<4} | {filename:<10} | {scores_dict['overall_score']:.4f} |", end=" ")
                criteria_scores = scores_dict['criteria_scores']
                for criterion in first_resume_scores:
                    print(f"{criteria_scores.get(criterion, 0):<12} |", end=" ")
                print()
                rank += 1
        else:
            print("\nNot enough data to train the machine learning model.")
    except Exception as e:
        print(f"Error during machine learning part (SVM): {e}")
