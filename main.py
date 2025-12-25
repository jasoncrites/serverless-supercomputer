import os
import json
from flask import Flask, request, jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, Part

app = Flask(__name__)

# Config
PROJECT_ID = os.environ.get("GCP_PROJECT", "afs-jasoncrites-1766633078257")
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Models
flash_model = GenerativeModel("gemini-1.5-flash-001")
pro_model = GenerativeModel("gemini-1.5-pro-001")

def analyze_complexity(prompt):
    """
    Uses Flash to cheaply determine if a prompt needs 'High IQ' or 'Fast Speed'.
    Returns: score (0-100)
    """
    analysis_prompt = f"""
    Analyze the complexity of this user request. 
    Score it 0-100 where 0 is "Hello/Simple Fact" and 100 is "Complex Reasoning/Coding/Math".
    Return ONLY the number.
    
    Request: {prompt[:500]}...
    """
    try:
        response = flash_model.generate_content(analysis_prompt)
        score = int(response.text.strip())
        return score
    except:
        return 50 # Default to medium

@app.route("/chat/completions", methods=["POST"])
def chat():
    """Simulate OpenAI API but routes smartly"""
    data = request.json
    prompt = data.get("messages", [])[-1].get("content", "")
    
    # 1. Smart Routing Analysis (Cost: negligible)
    complexity = analyze_complexity(prompt)
    
    # 2. Decision Logic (The Profit Layer)
    if complexity > 70:
        backend = "Gemini 1.5 Pro (High IQ)"
        # Use Pro model
        response = pro_model.generate_content(prompt)
    else:
        backend = "Gemini 1.5 Flash (Fast/Cheap)"
        # Use Flash model
        response = flash_model.generate_content(prompt)
        
    return jsonify({
        "choices": [{
            "message": {
                "content": response.text,
                "role": "assistant"
            }
        }],
        "model_used": backend,
        "complexity_score": complexity
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
