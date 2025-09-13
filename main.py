from http.client import HTTPException
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from deepface import DeepFace
import numpy as np

app = FastAPI(title="Face Verification Service")

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

# @app.post("/verify")
# async def verify_face(upload_image: UploadFile = File(...), db_image: UploadFile = File(...)):
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1, \
#              tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:

#             tmp1.write(await upload_image.read())
#             tmp2.write(await db_image.read())

#             result = DeepFace.verify(
#                 img1_path=tmp1.name,
#                 img2_path=tmp2.name,
#                 enforce_detection=False
#             )

#         return JSONResponse({"verified": result.get("verified", False)})
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/verify")
async def verify_face(
    db_image: UploadFile = File(...),       
    client_image: UploadFile = File(...),   
):
    """
    Compare DB image (from Django) with client-uploaded image using DeepFace.
    """

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as db_file:
            db_bytes = await db_image.read()
            np_arr = np.frombuffer(db_bytes, np.uint8)
            db_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imwrite(db_file.name, db_img)
            db_path = db_file.name

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as client_file:
            client_bytes = await client_image.read()
            client_file.write(client_bytes)
            client_path = client_file.name

        result = DeepFace.verify(
            img1_path=db_path,
            img2_path=client_path,
            model_name="Facenet",  # model choice can be adjusted
            detector_backend="opencv", # detector choice can be adjusted
            enforce_detection=False
        )

        return JSONResponse({
            "verified": result.get("verified"),
            "distance": result.get("distance"),
            "threshold": result.get("threshold"),
            "message": "Faces match ✅" if result.get("verified") else "Faces do not match"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")