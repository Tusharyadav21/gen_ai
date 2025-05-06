Generative AI
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install tiktoken

# Freeze the environment
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt

# Install OpenAI
pip install openai

# Install Python Environment
pip install python-dotenv

# Install Google Gemini
pip install google-genai