import tempfile
import subprocess
import os
import shutil
from fastapi import HTTPException
from app.core.config import settings
from app.services.s3_service import S3Services


async def process_pdf(s3_key: str):
    """
    Downloads a PDF from S3, converts it to markdown, and uploads the markdown file back to S3.

    Args:
        s3_key (str): S3 key of the PDF file.

    Returns:
        str: Local path of the generated markdown file.
    """
    s3_client = S3Services()

    output_directory = os.path.join("app", "static", "output")
    basename = os.path.splitext(os.path.basename(s3_key))[0]
    markdown_folder = os.path.join(output_directory, basename)
    markdown_file_path = os.path.join(markdown_folder, f"{basename}.md")
    markdown_file_name = f"{basename}.md"

    os.makedirs(markdown_folder, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_file_path = os.path.join(temp_dir, basename)
            s3_client.download_file(s3_key, pdf_file_path)

            output_directory = os.path.join("app", "static", "output")
            os.makedirs(output_directory, exist_ok=True)

            command = f"marker_single {pdf_file_path} {output_directory}"

            result = subprocess.run(command, shell=True,
                                    text=True, capture_output=True)

            if result.returncode != 0:
                raise Exception(f"Error in PDF conversion: {result.stderr}")

        s3_md_key = f"markdown/{markdown_file_name}"
        s3_client.upload_file(markdown_file_path, s3_md_key)

        return s3_md_key

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
