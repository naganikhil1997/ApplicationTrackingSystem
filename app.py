
from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from datetime import datetime
import fitz  # PyMuPDF
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import io
import re
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins by default

# Load environment variables
load_dotenv()
db_uri = os.getenv("MONGODB_URI")
db_name = 'local_pdf_db'
pdf_storage_path = 'C:\\Users\\user\\Desktop\\PAAP\\src\\assets\\uploads'
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Google AI
genai.configure(api_key=gemini_api_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# HTML conversion method
def get_html(text: str, simple: bool = False) -> str:
    def handle_links(line):
        link_pattern = re.compile(r'$(.*?)$$(.*?)$')
        line = link_pattern.sub(r'<a href="\2">\1</a>', line)
        
        url_pattern = re.compile(r'(http[s]?://[^\s]+)')
        line = url_pattern.sub(r'<a href="\1">\1</a>', line)
        
        email_pattern = re.compile(r'(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)')
        line = email_pattern.sub(r'<a href="mailto:\1">\1</a>', line)
        
        return line

    def escape_html(text):
        """Escape HTML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#039;"))

    lines = text.split('\n')
    
    if simple:
        html_output = ""
    else:
        html_output = """<div style="max-width: 1000px; padding: 15px; margin: 0 auto; height: 100%; display: flex; flex-direction: column; justify-content: center; overflow-x: auto;">"""
    
    list_open = False
    for line in lines:
        line = line.strip()
        if not line:
            html_output += '<p></p>'
            continue

        # Handle headers
        if line.startswith("# "):
            html_output += f'<h2>{escape_html(line[2:])}</h2>'
        elif line.startswith("## "):
            html_output += f'<h2>{escape_html(line[3:])}</h2>'
        elif line.startswith("### "):
            html_output += f'<h3>{escape_html(line[4:])}</h3>'
        elif line.startswith("**") and line.endswith("**"):
            html_output += f'<h2>{escape_html(line[2:-2])}</h2>'
        else:
            # Handle bullet points
            if line.startswith("* "):
                if not list_open:
                    html_output += '<ul>'
                    list_open = True
                html_output += f'<li>{escape_html(line[2:])}</li>'
            else:
                if list_open:
                    html_output += '</ul>'
                    list_open = False
                line = handle_links(escape_html(line))
                line = re.sub(r'\*(.*?)\*', r'<strong>\1</strong>', line)  # Bold
                line = re.sub(r'_(.*?)_', r'<em>\1</em>', line)  # Italics
                html_output += f'<div style="margin-bottom: 10px;">{line}</div>'

    if list_open:
        html_output += '</ul>'
    if not simple:
        html_output += '</div>'
    return html_output


# PDFProcessor class to handle PDF-related operations
class PDFProcessor:
    def __init__(self, db_uri, db_name):
        self.db_uri = db_uri
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        try:
            self.client = MongoClient(self.db_uri)
            self.db = self.client[self.db_name]
            self.pdf_collection = self.db['local_pdf_collection']
        except Exception as e:
            logging.error(f"Error connecting to MongoDB: {e}")
            raise

    def pdf_to_json(self, pdf_file):
        try:
            pdf_file.seek(0)
            logging.debug("Reading PDF file")
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            if doc.page_count == 0:
                raise Exception("The PDF file is empty or corrupted.")

            pdf_data = {"pages": []}
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                pdf_data["pages"].append({
                    "page_number": page_num + 1,
                    "text": text
                })
            logging.debug("PDF processing completed successfully")
            return pdf_data
        except Exception as e:
            logging.error(f"Error processing PDF: {e}")
            raise
    
    def extract_resume_details(self, resume_text):
        try:
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(f"Extract the name, email, and skills from the following resume:\n\n{resume_text}")
            result = response.text
            lines = result.split('\n')
            details = {
                "name": lines[0].strip() if len(lines) > 0 else "",
                "email": lines[1].strip() if len(lines) > 1 else "",
                "skills": lines[2].strip() if len(lines) > 2 else ""
            }
            return details
        except Exception as e:
            logging.error(f"Error extracting resume details: {e}")
            raise
    
    def process_pdf(self, file_name, pdf_file):
        try:
            file_path = os.path.join(pdf_storage_path, file_name)
            pdf_file.save(file_path)
            pdf_file.seek(0)
            json_data = self.pdf_to_json(pdf_file)
            if json_data is None:
                return

            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            page_count = len(doc)
            resume_text = ' '.join(page.get_text("text") for page in doc)

            details = self.extract_resume_details(resume_text)

            existing_record = self.pdf_collection.find_one({"file_name": file_name})

            if existing_record:
                self.pdf_collection.update_one(
                    {"file_name": file_name},
                    {"$set": {
                        "json_data": json_data,
                        "page_count": page_count,
                        "processed_date": datetime.utcnow(),
                        "file_path": file_path,
                        "details": details
                    }}
                )
                return {"status": "success", "message": f"Updated data for: {file_name}", "pdf_url": file_name}
            else:
                result = self.pdf_collection.insert_one({
                    "file_name": file_name,
                    "json_data": json_data,
                    "page_count": page_count,
                    "processed_date": datetime.utcnow(),
                    "file_path": file_path,
                    "details": details
                })
                return {"status": "success", "message": f"Processed and inserted: {file_name} with ID: {result.inserted_id}", "pdf_url": file_name}

        except Exception as e:
            logging.error(f"Error processing or saving PDF: {e}")
            raise

    def get_pdfs(self):
        try:
            pdfs = list(self.pdf_collection.find({}, {"_id": 0, "file_name": 1}))
            for pdf in pdfs:
                pdf['url'] = pdf['file_name']
            return pdfs
        except Exception as e:
            logging.error(f"Error fetching PDFs from MongoDB: {e}")
            raise

    def get_pdf_content(self, file_name):
        try:
            document = self.pdf_collection.find_one({"file_name": file_name})
            if document:
                return document.get("json_data", {})
            else:
                return {}
        except Exception as e:
            logging.error(f"Error retrieving PDF content: {e}")
            raise

    def get_pdf_bytes(self, file_name):
        try:
            document = self.pdf_collection.find_one({"file_name": file_name})
            if document:
                file_path = document.get("file_path", None)
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        return f.read()
            return None
        except Exception as e:
            logging.error(f"Error retrieving PDF file data: {e}")
            raise

    def match_resumes(self, job_description):
        try:
            resumes = list(self.pdf_collection.find({}))
            matched_resumes = []

            for resume in resumes:
                file_name = resume.get("file_name")
                content = resume.get("json_data", {}).get("pages", [])
                resume_text = ' '.join(page.get("text", "") for page in content)

                # Start an AI chat session to evaluate the resume against the job description
                chat_session = model.start_chat(history=[])
                prompt = (
                    f"You are an expert in recruitment. Evaluate this resume and determine how well it matches the following job description:\n\n"
                    f"Job Description:\n{job_description}\n\nResume:\n{resume_text}\n\n"
                    f"Provide a percentage score indicating how well the resume aligns with the job description, and briefly explain your reasoning."
                )
                response = chat_session.send_message(prompt)
                ai_feedback = response.text

                # Extract match score from AI feedback
                score_match = re.search(r'Score: (\d+)%', ai_feedback)
                score = int(score_match.group(1)) if score_match else 0

                # Append resume details if the score is strong enough (e.g., above 70%)
                if score >= 70: 
                    matched_resumes.append({
                        "file_name": file_name,
                        "score": score,
                        "ai_feedback": ai_feedback.strip()
                    })

            return matched_resumes

        except Exception as e:
            logging.error(f"Error matching resumes: {e}")
            raise


# Initialize the PDF processor
processor = PDFProcessor(db_uri, db_name)

@app.route('/ai-resume-query', methods=['POST'])
def ai_resume_query():
    try:
        data = request.json
        user_query = data.get("query", "").lower()

        if not user_query:
            return jsonify({"status": "error", "message": "Query is required"}), 400

        restricted_keywords = ["show", "list", "download", "view", "get", "give", "access"]
        restricted_objects = ["resume", "resumes", "cv", "file", "documents"]

        if any(keyword in user_query for keyword in restricted_keywords) and any(obj in user_query for obj in restricted_objects):
            return """
            <div style="color: red; font-weight: bold;">
                To view or download resumes, please complete the payment process.
            </div>
            """, 403

        resumes = list(processor.pdf_collection.find({}))

        all_resumes_text = []
        for resume in resumes:
            file_name = resume.get("file_name")
            content = resume.get("json_data", {}).get("pages", [])
            resume_text = ' '.join(page.get("text", "") for page in content)
            all_resumes_text.append(f"Resume: {file_name}\n{resume_text}\n")

        resumes_combined_text = '\n'.join(all_resumes_text)

        chat_session = model.start_chat(history=[])
        prompt = f"The user has asked the following query related to resumes:\n\n{user_query}\n\nHere are the resumes from the database:\n{resumes_combined_text}\n\nPlease provide a detailed response."
        response = chat_session.send_message(prompt)
        ai_response = response.text

        # Convert the AI response to HTML format
        html_response = get_html(ai_response)
        return html_response, 200

    except Exception as e:
        logging.error(f"Error in /ai-resume-query endpoint: {e}")
        return f"""
        <div style="color: red; font-weight: bold;">
            Error: {str(e)}
        </div>
        """, 500


@app.route('/ai-resume-upload-query', methods=['POST'])
def ai_resume_upload_query():
    try:
        if 'file' not in request.files or 'query' not in request.form:
            return jsonify({"status": "error", "message": "Resume file and query are required"}), 400

        file = request.files['file']
        user_query = request.form['query'].lower()

        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        pdf_processor = PDFProcessor(db_uri, db_name)

        pdf_file = file
        resume_text = ' '.join(page['text'] for page in pdf_processor.pdf_to_json(pdf_file)["pages"])

        chat_session = model.start_chat(history=[])

        prompt = (
            f"An HR manager is reviewing a resume and has the following question: {user_query}. "
            f"Here is the content of the resume:\n\n{resume_text}\n\n"
            f"Please answer the HR manager's query based on the resume information."
        )

        response = chat_session.send_message(prompt)
        ai_response = response.text

        # Convert the AI response to HTML format
        html_response = get_html(ai_response)
        return html_response, 200

    except Exception as e:
        logging.error(f"Error in /ai-resume-upload-query endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_resumes():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    file_name = file.filename
    try:
        result = processor.process_pdf(file_name, file)
        pdf_url = request.host_url + 'pdf/' + file_name
        pdf_content = processor.get_pdf_content(file_name)
        result['pdf_url'] = pdf_url
        result['pdf_content'] = pdf_content
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in /upload endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/pdfs', methods=['GET'])
def get_pdfs():
    try:
        pdfs = processor.get_pdfs()
        return jsonify(pdfs)
    except Exception as e:
        logging.error(f"Error in /pdfs endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/pdf/<file_name>', methods=['GET'])
def get_pdf(file_name):
    try:
        pdf_bytes = processor.get_pdf_bytes(file_name)
        if pdf_bytes:
            return send_file(io.BytesIO(pdf_bytes), as_attachment=True, download_name=file_name)
        else:
            return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        logging.error(f"Error in /pdf/<file_name> endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/job-matching', methods=['POST'])
def match_resumes():
    data = request.json
    job_description = data.get("job_description", "")

    if not job_description:
        return jsonify({"status": "error", "message": "Job description is required"}), 400

    try:
        matched_resumes = processor.match_resumes(job_description)
        return jsonify(matched_resumes)
    except Exception as e:
        logging.error(f"Error in /match endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/evaluate-job-posting', methods=['POST'])
def evaluate_job_posting():
    try:
        data = request.json
        job_title = data.get("job_title", "").lower()
        job_description = data.get("job_description", "").lower()

        if not job_title or not job_description:
            return jsonify({"status": "error", "message": "Job title and description are required"}), 400

        job_full_description = f"{job_title} {job_description}"

        resumes = list(processor.pdf_collection.find({}))

        if not resumes:
            logging.info("No resumes available for evaluation.")
            return jsonify({"status": "success", "message": "No resumes available for evaluation", "data": []}), 200

        matched_resumes = []

        for resume in resumes:
            file_name = resume.get("file_name")
            content = resume.get("json_data", {}).get("pages", [])

            # Combine all pages of the resume into a single text block
            resume_text = ' '.join(page.get("text", "").lower() for page in content)

            # Use AI to evaluate the resume against the job description
            chat_session = model.start_chat(history=[])
            prompt = (
                f"You are an expert ATS (Applicant Tracking System) scanner. Analyze the following resume in the context of the job description.\n\n"
                f"Job Description:\n{job_full_description}\n\nResume:\n{resume_text}\n\n"
                f"Provide a percentage score from 0% to 100% on how well this resume matches the job requirements, and explain your reasoning."
            )
            response = chat_session.send_message(prompt)
            ai_feedback = response.text

            # Extract match score from AI feedback
            score_match = re.search(r'Score: (\d+)%', ai_feedback)
            score = int(score_match.group(1)) if score_match else 0

            # Add the resume to the matched resumes if the score is high enough (e.g., above 70%)
            if score >= 70:
                matched_resumes.append({
                    "file_name": file_name,
                    "match_score": score,
                    "ai_feedback": ai_feedback.strip()
                })

        # Build a full HTML response to include all matched resumes
        html_output = """<div style="max-width: 1000px; padding: 15px; margin: 0 auto; height: 100%; display: flex; flex-direction: column; justify-content: center; overflow-x: auto;">"""

        for resume in matched_resumes:
            html_output += f"""
            <div style="margin-bottom: 20px; border: 1px solid #ccc; padding: 10px;">
                <h3>File Name: {resume['file_name']}</h3>
                <p>Match Score: {resume['match_score']:.2f}%</p>
                {resume['ai_feedback']}
            </div>
            """
        
        html_output += '</div>'

        return html_output, 200

    except Exception as e:
        logging.error(f"Error in /evaluate-job-posting endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)