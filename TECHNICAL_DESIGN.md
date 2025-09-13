# Technical Design Document: Capstone Application

## Overview
The Capstone application is a multilingual text and speech processing tool that leverages Google Cloud and Gemini AI services. It provides a user-friendly web interface for translating text, transcribing audio, and generating speech audio. The application is structured into three main Python modules, each with a distinct responsibility:

- **UI Layer:** `streamlit_app.py`
- **Backend Layer:** `services.py`
- **Utilities Layer:** `audio_utils.py`

---

## 1. UI Layer: `streamlit_app.py`

### Purpose
This file implements the user interface using Streamlit, enabling users to interact with the application through a web browser. It orchestrates the workflow from user input to translation and audio output.

### Key Responsibilities
- Page configuration and layout using Streamlit.
- Collecting user input via text, file upload, or microphone.
- Displaying results: transcribed text, translated text, and generated audio.
- Managing session state for a seamless user experience.
- Handling errors and providing user guidance.

### Significance
`streamlit_app.py` acts as the entry point for users, abstracting backend complexity and providing a simple, interactive experience. It bridges user actions with backend services and utility functions.

---

## 2. Backend Layer: `services.py`

### Purpose
This module encapsulates the core business logic and integration with external AI services (Google Gemini, Google Speech-to-Text, gTTS, etc.).

### Key Responsibilities
- Initializing and managing API clients for Gemini and Google Cloud.
- Translating text using Gemini generative models.
- Converting text to speech using gTTS.
- Transcribing audio using Google Speech-to-Text, with robust handling for multiple audio formats.
- Extracting text from uploaded files (TXT, PDF, CSV, Excel).
- Providing error handling and fallback mechanisms for service calls.

### Significance
`services.py` centralizes all interactions with external services, ensuring modularity and maintainability. It allows the UI to remain agnostic of service implementation details, supporting scalability and easier updates.

---

## 3. Utilities Layer: `audio_utils.py`

### Purpose
This module provides low-level audio processing utilities and format detection functions, supporting the backend and UI layers.

### Key Responsibilities
- Detecting audio file formats (WAV, OGG, WEBM) from byte streams.
- Extracting sample rates from WAV files.
- Wrapping microphone recording functionality for safe use in Streamlit.
- Providing reusable, robust helpers for audio data validation and processing.

### Significance
`audio_utils.py` abstracts audio format handling and validation, reducing code duplication and potential errors. It enables the backend to process diverse audio inputs reliably, enhancing the application's robustness.

---

## Conclusion
The Capstone application's modular design—separating UI, backend, and utility concerns—ensures clarity, maintainability, and extensibility. Each Python file plays a critical role: the UI delivers a seamless user experience, the backend manages business logic and service integration, and the utilities provide foundational audio processing support. This architecture enables rapid development and easy adaptation to new requirements or technologies.
