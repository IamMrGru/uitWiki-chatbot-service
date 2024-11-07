from fastapi import HTTPException
import tempfile
from app.core.config import settings
import subprocess
import boto3
import os
import shutil

s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)


async def process_pdf(s3_key: str):
    """
    Downloads a PDF from S3, converts it to markdown, and uploads the markdown file back to S3.

    Args:
        s3_key (str): S3 key of the PDF file.

    Returns:
        str: Local path of the generated markdown file.
    """
    bucket_name = settings.S3_BUCKET_NAME
    output_directory = os.path.join("app", "static", "output")
    basename = os.path.splitext(os.path.basename(s3_key))[0]
    markdown_folder = os.path.join(output_directory, basename)
    markdown_file_path = os.path.join(markdown_folder, f"{basename}.md")
    markdown_file_name = f"{basename}.md"

    os.makedirs(markdown_folder, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_file_path = os.path.join(temp_dir, basename)
            s3_client.download_file(bucket_name, s3_key, pdf_file_path)

            output_directory = os.path.join("app", "static", "output")
            os.makedirs(output_directory, exist_ok=True)

            command = f"marker_single {pdf_file_path} {output_directory}"

            result = subprocess.run(command, shell=True,
                                    text=True, capture_output=True)

            if result.returncode != 0:
                raise Exception(f"Error in PDF conversion: {result.stderr}")

        s3_md_key = f"markdown/{markdown_file_name}"
        s3_client.upload_file(markdown_file_path, bucket_name, s3_md_key)

        return markdown_file_path

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{str(e)}")

    finally:
        if os.path.exists(pdf_file_path):
            os.remove(pdf_file_path)

        output_dir = os.path.join(
            "app", "static", "output", basename)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
