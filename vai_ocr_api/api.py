from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import requests
import logging
from typing import List
import io

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.put("/id_card/")
async def process_id_card(
    image_front: UploadFile = File(...),
    image_back: UploadFile = File(...)
):
    try:
        logger.info("Processing ID card images")
        
        # Read file contents
        front_content = await image_front.read()
        back_content = await image_back.read()
        
        # Prepare the files for the API request
        files = [
            ('image_front', ('front.jpg', front_content, 'image/jpeg')),
            ('image_back', ('back.jpg', back_content, 'image/jpeg'))
        ]
        
        # Make request to Viettel AI API
        url = "https://viettelai.vn/ocr/id_card"
        response = requests.post(
            url=url,
            files=files,
            headers={},
            data={}
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Return the API response
        return JSONResponse(content={
            "status": "success",
            "data": response.json(),
            "metadata": {
                "front_image": {
                    "filename": image_front.filename,
                    "content_type": image_front.content_type,
                    "size": len(front_content)
                },
                "back_image": {
                    "filename": image_back.filename,
                    "content_type": image_back.content_type,
                    "size": len(back_content)
                }
            }
        })

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Viettel AI API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling external API: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing ID card images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=39000)