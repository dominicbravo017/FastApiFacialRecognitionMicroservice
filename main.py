from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from deepface import DeepFace

app = FastAPI(title="Face Verification Service")

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
