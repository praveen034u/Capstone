# create/activate venv
python -m venv .venv
.\.venv\Scripts\activate

# install deps
pip install -r requirements.txt

# (local) put a test key in .streamlit/secrets.toml

# run
streamlit run translate_tts_app.py