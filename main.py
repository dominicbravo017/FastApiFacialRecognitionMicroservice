# import os
# import cv2
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import tempfile
# from deepface import DeepFace
# import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# app = FastAPI(title="Face Verification Service")

# @app.get("/")
# def health():
#     return {"status": "ok"}

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000)) 
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

# @app.post("/verify")
# async def verify_face(upload_image: UploadFile = File(...), db_image: UploadFile = File(...)):
#     tmp1 = tmp2 = None
#     try:
#         tmp1 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
#         tmp2 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

#         upload_bytes = await upload_image.read()
#         db_bytes = await db_image.read()

#         img1 = cv2.imdecode(np.frombuffer(upload_bytes, np.uint8), cv2.IMREAD_COLOR)
#         img2 = cv2.imdecode(np.frombuffer(db_bytes, np.uint8), cv2.IMREAD_COLOR)

#         img1_resized = cv2.resize(img1, (160, 160))
#         img2_resized = cv2.resize(img2, (160, 160))

#         cv2.imwrite(tmp1.name, img1_resized)
#         cv2.imwrite(tmp2.name, img2_resized)

#         result = DeepFace.verify(
#             img1_path=tmp1.name,
#             img2_path=tmp2.name,
#             model_name="Facenet",
#             enforce_detection=False
#         )

#         return JSONResponse({
#             "verified": result.get("verified"),
#             "distance": result.get("distance"),
#             "threshold": result.get("threshold"),
#             "message": "Faces match ✅" if result.get("verified") else "Faces do not match ❌"
#         })

#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)

#     finally:
#         if tmp1:
#             tmp1.close()
#             os.remove(tmp1.name)
#         if tmp2:
#             tmp2.close()
#             os.remove(tmp2.name)



# # @app.post("/verify")
# # async def verify_face(
# #     db_image: UploadFile = File(...),       
# #     client_image: UploadFile = File(...),   
# # ):
# #     """
# #     Compare DB image (from Django) with client-uploaded image using DeepFace.
# #     """
# #     try:
# #         # Save DB image
# #         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as db_file:
# #             db_bytes = await db_image.read()
# #             np_arr = np.frombuffer(db_bytes, np.uint8)
# #             db_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
# #             cv2.imwrite(db_file.name, db_img)
# #             db_path = db_file.name

# #         # Save Client image
# #         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as client_file:
# #             client_bytes = await client_image.read()
# #             client_file.write(client_bytes)
# #             client_path = client_file.name

# #         # Run DeepFace verification
# #         result = DeepFace.verify(
# #             img1_path=db_path,
# #             img2_path=client_path,
# #             enforce_detection=False
# #         )

# #         # Clean up
# #         os.remove(db_path)
# #         os.remove(client_path)

# #         return JSONResponse({
# #             "verified": result.get("verified"),
# #             "distance": result.get("distance"),
# #             "threshold": result.get("threshold"),
# #             "message": "Faces match ✅" if result.get("verified") else "Faces do not match"
# #         })

# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")

import os
import cv2
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
from contextlib import asynccontextmanager
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Path to your bundled weights (inside project/models/)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "facenet_weights.h5")

# DeepFace cache folder (~/.deepface/weights/)
DEEFPACE_CACHE = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
DEEFPACE_MODEL_PATH = os.path.join(DEEFPACE_CACHE, "facenet_weights.h5")

facenet_model = None  # global reference


@asynccontextmanager
async def lifespan(app: FastAPI):
    global facenet_model
    try:
        # Ensure DeepFace weights folder exists
        os.makedirs(DEEFPACE_CACHE, exist_ok=True)

        # Copy your local model file if not already cached
        if os.path.exists(MODEL_PATH) and not os.path.exists(DEEFPACE_MODEL_PATH):
            shutil.copy(MODEL_PATH, DEEFPACE_MODEL_PATH)
            print(f"✅ Copied local weights to {DEEFPACE_MODEL_PATH}")
        elif os.path.exists(DEEFPACE_MODEL_PATH):
            print("✅ Found cached Facenet weights")
        else:
            print("⚠️ No local weights found, DeepFace may try downloading...")

        # Build the Facenet model (DeepFace will auto-load weights)
        facenet_model = DeepFace.build_model("Facenet")
        print("✅ Facenet model initialized")

    except Exception as e:
        print(f"❌ Error initializing Facenet model: {e}")

    yield  # Run app

    # Cleanup on shutdown
    facenet_model = None
    print("🛑 Facenet model unloaded")


app = FastAPI(title="Face Verification Service", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/verify")
async def verify_face(upload_image: UploadFile = File(...), db_image: UploadFile = File(...)):
    tmp1 = tmp2 = None
    try:
        tmp1 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp2 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

        upload_bytes = await upload_image.read()
        db_bytes = await db_image.read()

        img1 = cv2.imdecode(np.frombuffer(upload_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(db_bytes, np.uint8), cv2.IMREAD_COLOR)

        img1_resized = cv2.resize(img1, (160, 160))
        img2_resized = cv2.resize(img2, (160, 160))

        cv2.imwrite(tmp1.name, img1_resized)
        cv2.imwrite(tmp2.name, img2_resized)

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
        if tmp1:
            tmp1.close()
            os.remove(tmp1.name)
        if tmp2:
            tmp2.close()
            os.remove(tmp2.name)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
