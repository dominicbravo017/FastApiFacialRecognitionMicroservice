import os
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from deepface import DeepFace
import numpy as np

# ✅ Force CPU only (ignore GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI(title="Face Verification Service")

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

@app.post("/verify")
async def verify_face(upload_image: UploadFile = File(...), db_image: UploadFile = File(...)):
    tmp1 = tmp2 = None
    try:
        # Create temp files
        tmp1 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp2 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

        # Read uploaded images
        upload_bytes = await upload_image.read()
        db_bytes = await db_image.read()

        # Decode images
        img1 = cv2.imdecode(np.frombuffer(upload_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(db_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Resize images to 160x160 (for Facenet)
        img1_resized = cv2.resize(img1, (160, 160))
        img2_resized = cv2.resize(img2, (160, 160))

        # Save resized images temporarily
        cv2.imwrite(tmp1.name, img1_resized)
        cv2.imwrite(tmp2.name, img2_resized)

        # Run DeepFace verification
        result = DeepFace.verify(
            img1_path=tmp1.name,
            img2_path=tmp2.name,
            model_name="Facenet",
            enforce_detection=False
        )

        return JSONResponse({
            "verified": result.get("verified"),
            "distance": result.get("distance"),
            "threshold": result.get("threshold"),
            "message": "Faces match ✅" if result.get("verified") else "Faces do not match ❌"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        # Clean up temporary files
        if tmp1:
            tmp1.close()
            os.remove(tmp1.name)
        if tmp2:
            tmp2.close()
            os.remove(tmp2.name)



# @app.post("/verify")
# async def verify_face(
#     db_image: UploadFile = File(...),       
#     client_image: UploadFile = File(...),   
# ):
#     """
#     Compare DB image (from Django) with client-uploaded image using DeepFace.
#     """
#     try:
#         # Save DB image
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as db_file:
#             db_bytes = await db_image.read()
#             np_arr = np.frombuffer(db_bytes, np.uint8)
#             db_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             cv2.imwrite(db_file.name, db_img)
#             db_path = db_file.name

#         # Save Client image
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as client_file:
#             client_bytes = await client_image.read()
#             client_file.write(client_bytes)
#             client_path = client_file.name

#         # Run DeepFace verification
#         result = DeepFace.verify(
#             img1_path=db_path,
#             img2_path=client_path,
#             enforce_detection=False
#         )

#         # Clean up
#         os.remove(db_path)
#         os.remove(client_path)

#         return JSONResponse({
#             "verified": result.get("verified"),
#             "distance": result.get("distance"),
#             "threshold": result.get("threshold"),
#             "message": "Faces match ✅" if result.get("verified") else "Faces do not match"
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")
