# Stride-AI
The app for adventurous runners, walkers, and hikers.

## Requirements
Copy `.env.example` to `.env` and fill in the required API keys from the following sources:

### Groq API Key
https://console.groq.com/keys

### Open Route Service API Key
https://api.openrouteservice.org -> https://account.heigit.org/manage/key

## Program Usage
Create a virtual environment and install the requirements with:
```
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

Run the program with:
```bash
./venv/bin/python agent.py
```

## Viewing Generated Routes
Generated routes are saved in `runs/` as `YYYYMMDD_HHMMSS_map.html` and are viewable in a web browser.

