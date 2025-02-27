import boto3

from app.core.config import settings


class S3Services:
    def __init__(self):
        self.aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        self.aws_region = settings.AWS_REGION
        self.bucket_name = settings.S3_BUCKET_NAME
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )

    def download_file(self, s3_key: str, file_path: str):
        print(f"Downloading file from S3: {s3_key}")
        self.s3.download_file(self.bucket_name, s3_key, file_path)

    def upload_file(self,  file_path: str, s3_key: str):
        self.s3.upload_file(file_path, self.bucket_name, s3_key)
