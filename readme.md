# 📄 For a detailed technical overview of this project, please see [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md).


#How to run this app in local-
# create/activate venv
python -m venv .venv
.\.venv\Scripts\activate

# install deps
pip install -r requirements.txt

# (local) put a test key in .streamlit/secrets.toml

# run
streamlit run translate_tts_app.py