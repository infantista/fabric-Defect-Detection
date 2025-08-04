import gdown
import zipfile
import os

# Google Drive file ID and download URL
file_id = '1ki3KugioGyMzXpT8zhbGl_5DuyEV4DcJ'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'data.zip'

print("⬇️ Downloading dataset from Google Drive...")
gdown.download(url, output, quiet=False)

print("📦 Extracting ZIP file...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('fabric-Defect-Detection/data')

os.remove(output)
print("✅ Dataset ready in 'fabric-Defect-Detection/data'")
