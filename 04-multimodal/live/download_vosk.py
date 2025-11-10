import requests
import zipfile
import os
from pathlib import Path

url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
zip_path = "vosk-ru.zip"
extract_path = "models/vosk-ru-small"

print(f"Downloading {url}...")
response = requests.get(url, stream=True)
response.raise_for_status()

with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Download complete. Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("models")

# Rename the extracted folder
extracted_folder = "models/vosk-model-small-ru-0.22"
if os.path.exists(extracted_folder):
    os.rename(extracted_folder, extract_path)

print("Extraction complete.")
os.remove(zip_path)
print("Cleaned up zip file.")
