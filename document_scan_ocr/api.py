from fastapi import FastAPI, UploadFile, File
import os
import torch
from PIL import Image
import shutil
import base64
from openai import OpenAI
from typing import Dict
import tempfile
import re
import uvicorn
from pdf2image import convert_from_path
from transformers import MllamaForConditionalGeneration, AutoProcessor

app = FastAPI()

# Cấu hình CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Khởi tạo model Llama
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llama_model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).to(device)
llama_processor = AutoProcessor.from_pretrained(model_id)

# OpenAI config
GPT_MODEL = 'chatgpt-4o-latest'
OPENAI_API_KEY = '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'

def convert_pdf_to_images(pdf_path) -> list:
    output_folder = 'images'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    images = convert_from_path(pdf_path)
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i+1}.jpg')
        image.save(image_path, 'JPEG')
        image_paths.append(image_path)
    
    return image_paths

def get_image_base64(image_path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_with_llama(image_path) -> str:
    image = Image.open(image_path)
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": 
            """Act as an OCR assistant and table extractor. Extract information with this step:
1. Recognize all visible text in the image **accurately** in Vietnamese.
2. Correct spelling mistakes or words you think don't make sense if necessary.
3. If the image contains paragraphs, extract all this with format plain text and enclosed in parentheses by the tag pair <ptext> </ptext>.
4. If the image contains tables, extract all tabular data into a structured **JSON format** and enclosed all information extracted in parentheses by the tag pair <board> </board>. 
If table is not exist in image, please don't generate <board> and </board> (Avoid information distortion and redundancy).
5. Do not provide any additional explanations, Just focus on getting all the text in the image, no missing or adding anything.
"""}
        ]}
    ]
    
    input_text = llama_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = llama_processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(llama_model.device)

    output = llama_model.generate(**inputs, max_new_tokens=2000, num_beams=3)
    return llama_processor.decode(output[0])

def extract_text_from_image(base64_image) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": """Act as an OCR assistant and table extractor. Extract information with this step:
1. Recognize all visible text in the image **accurately** in Vietnamese.
2. Correct spelling mistakes or words you think don't make sense if necessary.
3. If the image contains paragraphs, extract all this with format plain text and enclosed in parentheses by the tag pair <ptext> </ptext>.
4. If the image contains tables, extract all tabular data into a structured **JSON format** and enclosed all information extracted in parentheses by the tag pair <board> </board>. 
If table is not exist in image, please don't generate <board> and </board> (Avoid information distortion and redundancy).
5. Do not provide any additional explanations"""},
            {"role": "user", "content": [
                {'type': 'text', 'text': 'Trích xuất cho tôi đầy đủ thông tin của bức ảnh sau'},
                {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
    )
    return response.choices[0].message.content

def combine_extracted_content(contents) -> str:
    final_output = []
    
    for page_num, content in enumerate(contents, 1):
        text_pattern = r'<ptext>.*?</ptext>'
        board_pattern = r'<board>.*?</board>'
        
        text_matches = [(m.group(), m.start()) for m in re.finditer(text_pattern, content, re.DOTALL)]
        board_matches = [(m.group(), m.start()) for m in re.finditer(board_pattern, content, re.DOTALL)]
        
        all_matches = text_matches + board_matches
        all_matches.sort(key=lambda x: x[1])
        
        page_content = [match[0] for match in all_matches]
        
        if page_content:
            final_output.extend(page_content)

    final_output = '\n'.join(final_output)
    final_output = final_output.replace("<ptext> </ptext>","").replace("<board> </board>","").replace("<board> and </board>","")

    return final_output

@app.put("/extract-pdf-content-llama/")
async def extract_pdf_content_llama(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        image_paths = convert_pdf_to_images(temp_pdf_path)
        
        extracted_contents = []
        for image_path in image_paths:
            extracted_text = extract_text_with_llama(image_path)
            extracted_contents.append(extracted_text)
        
        final_content = combine_extracted_content(extracted_contents)
        
        os.unlink(temp_pdf_path)
        shutil.rmtree('images')
        
        return {"extracted_content": final_content}
        
    except Exception as e:
        return {"error": str(e)}

@app.put("/extract-pdf-content-gpt/")
async def extract_pdf_content_gpt(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

        image_paths = convert_pdf_to_images(temp_pdf_path)
        
        extracted_contents = []
        for image_path in image_paths:
            base64_image = get_image_base64(image_path)
            extracted_text = extract_text_from_image(base64_image)
            extracted_contents.append(extracted_text)
        
        final_content = combine_extracted_content(extracted_contents)
        
        os.unlink(temp_pdf_path)
        shutil.rmtree('images')
        
        return {"extracted_content": final_content}
        
    except Exception as e:
        return {"error": str(e)}
        
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7012, reload=True)
