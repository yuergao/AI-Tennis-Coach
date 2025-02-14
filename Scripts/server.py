# load model
from eval import get_segments
# create http endpoint to get prediction for a given video

import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uuid

app = FastAPI()
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

@app.post("/segment/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    content_length = request.headers.get("content-length")
    print(content_length)
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Prevent directory traversal attacks
    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    from pathlib import Path
    file_ext = Path(file.filename).suffix
    if not file_ext == ".mp4":
        raise HTTPException(status_code=400, detail="Please send mp4 file")
    
    unique_filename = f"temp_{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    # Save the file securely
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    segments = get_segments(file_path)
    return {"segments": segments, "status": "Segmented successfully"}

# get your favorite player from the list of players, and sync

if __name__ == "__main__":
    '''
    export CPU_ONLY=true
    uvicorn server:app
    added basic security measures like preventing directory traversal attacks, allowing only mp4, and limiting file size
    testing the segment endpoint using curl
    curl -X 'POST' \
        'http://127.0.0.1:8000/segment/' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F 'file=@Annotations/videos/Amateur/insta1.mp4'
    '''
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
