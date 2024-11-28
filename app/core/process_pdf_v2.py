import base64
import os
from io import BytesIO

import pandas as pd
import requests
from langchain_core.messages import HumanMessage
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from openai import OpenAI
from pdf2image import convert_from_bytes
from pydantic import SecretStr
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm

from app.core.config import settings
from app.services.pinecone_service import PineconeService

oai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro', api_key=SecretStr(settings.GOOGLE_API_KEY))


def convert_page_to_image(pdf_bytes, page_number):
    images = convert_from_bytes(pdf_bytes)

    image = images[0]

    images_dir = 'app/static/images'

    os.makedirs(images_dir, exist_ok=True)

    image_file_name = f"page_{page_number}.png"
    image_file_path = os.path.join(images_dir, image_file_name)
    image.save(image_file_path, 'png')

    return image_file_path


def chunk_document(document_url):
    response = requests.get(document_url)

    pdf_data = response.content

    pdf_reader = PdfReader(BytesIO(pdf_data))
    page_chunks = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)
        pdf_bytes_io = BytesIO()
        pdf_writer.write(pdf_bytes_io)
        pdf_bytes_io.seek(0)
        pdf_bytes = pdf_bytes_io.read()
        page_chunk = {
            'pageNumber': page_number,
            'pdfBytes': pdf_bytes
        }
        page_chunks.append(page_chunk)

    return page_chunks


def encode_image(local_image_path):
    with open(local_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_vision_response(prompt, image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    )

    ai_msg = llm.invoke([message])

    return ai_msg


def process_document(document_url):
    try:
        print("Document processing started")

        page_chunks = chunk_document(document_url)

        total_pages = len(page_chunks)

        page_data_list = []

        for page_chunk in tqdm(page_chunks, total=total_pages, desc='Processing Pages'):
            page_number = page_chunk['pageNumber']
            pdf_bytes = page_chunk['pdfBytes']

            image_path = convert_page_to_image(pdf_bytes, page_number)

            system_prompt = (
                "Người dùng sẽ cung cấp cho bạn một hình ảnh của tệp tài liệu. Thực hiện các hành động sau: "
                "1. Chép lại văn bản trên trang thành một cấu trúc đẹp dưới dạng Markdown. **BẢN CHÉP LẠI VĂN BẢN:** "
                "2. Nếu có biểu đồ, mô tả hình ảnh và bao gồm văn bản **MÔ TẢ HÌNH ẢNH HOẶC BIỂU ĐỒ:** "
                "3. Nếu có bảng, chép lại bảng và bao gồm văn bản trên trang thành một cấu trúc đẹp dưới dạng Markdown **BẢN CHÉP LẠI BẢNG:**"
            )

            vision_response = get_vision_response(system_prompt, image_path)

            text = vision_response.content

            page_data = {
                'PageNumber': page_number,
                'ImagePath': image_path,
                'PageText': text
            }
            page_data_list.append(page_data)

        pdf_df = pd.DataFrame(page_data_list)
        print("Document processing completed.")
        print("DataFrame created with page data.")

        return pdf_df

    except Exception as err:
        print(f"Error processing document: {err}")


def get_embedding(text_input):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=SecretStr(settings.GOOGLE_API_KEY))

    return embeddings.embed_query(text_input)


pinecone_service = PineconeService()


async def upsert_vector(identifier, metadata):
    try:
        await pinecone_service.upsert_chunk({
            'id': identifier,
            'metadata': metadata,
            'text': metadata['text']
        })
    except Exception as e:
        print(f"Error upserting vector with ID {identifier}: {e}")
        raise
