from flask import Flask, render_template, request
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# Function to convert GitHub URL to raw URL
def convert_to_raw_url(github_url):
    base_url = "https://raw.githubusercontent.com"
    parts = github_url.split("/")
    repo_path = "/".join(parts[3:])  # Get the repo path after the domain
    raw_url = f"{base_url}/{repo_path}"
    return raw_url

# Function to fetch code from GitHub raw URL
def get_code_from_github(raw_url):
    try:
        response = requests.get(raw_url)
        if response.status_code == 200:
            return response.text
        else:
            return "Error: Could not fetch the code."
    except Exception as e:
        return f"Error: {str(e)}"

# LangChain setup for code analysis
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key="AIzaSyCpBudmGSNtIo6pR3famxbkpSEXpYKOZDY"
)

# Prompt for code analysis
analysis_template = """
Below is a code snippet. Analyze it strictly and return only:
1. The general purpose and context of the code.
2. Any errors or potential issues in the code. If there are no errors, explicitly state "No errors found." 

Please do not provide any other information or the code itself.

Code:
{code}
"""
analysis_prompt = PromptTemplate.from_template(analysis_template)

# Prompt for code correction
correction_template = """
Below is a code snippet. If there are errors in the code, provide the corrected version with very short comment of where the code is corrected.
If there are no errors, return the code as is without any additional explanation or comments.

Code:
{code}
"""
correction_prompt = PromptTemplate.from_template(correction_template)

# Function to analyze code using LangChain
def analyze_code_with_llm(code):
    chain = analysis_prompt | llm
    response = chain.invoke({"code": code})
    return response.content.strip()

# Function to correct code using LangChain
def correct_code_with_llm(code):
    chain = correction_prompt | llm
    response = chain.invoke({"code": code})
    return response.content.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract_code', methods=['POST'])
def extract_code():
    github_url = request.form['github_url']
    raw_url = convert_to_raw_url(github_url)
    code = get_code_from_github(raw_url)
    return render_template('index.html', code=code)

@app.route('/analyze_code', methods=['POST'])
def analyze_code():
    code = request.form['code']
    # Perform analysis using LangChain
    analysis_result = analyze_code_with_llm(code)
    return render_template('analysis.html', code=code, analysis_result=analysis_result)

@app.route('/correct_code', methods=['POST'])
def correct_code():
    code = request.form['code']
    # Perform code correction using LangChain
    corrected_code = correct_code_with_llm(code)
    return render_template('corrected_code.html', corrected_code=corrected_code)

if __name__ == "__main__":
    app.run(debug=True)
