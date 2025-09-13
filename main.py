import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from deepface import DeepFace

app = FastAPI(title="Face Verification Service")

@app.get("/")
def health():
    return {"status": "ok"}

# Only runs locally (not on Render's web command)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # use Render's port
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

@app.post("/verify")
async def verify_face(upload_image: UploadFile = File(...), db_image: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:

            tmp1.write(await upload_image.read())
            tmp2.write(await db_image.read())

            result = DeepFace.verify(
                img1_path=tmp1.name,
                img2_path=tmp2.name,
                enforce_detection=False
            )

        return JSONResponse({"verified": result.get("verified", False)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
