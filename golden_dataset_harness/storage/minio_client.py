from __future__ import annotations

import io
import asyncio
from minio import Minio

class ObjectStorageClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket = bucket

    def ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    async def upload_image(self, image_id: str, image_bytes: bytes, content_type: str = 'image/jpeg') -> str:
        def _upload():
            self.client.put_object(
                self.bucket,
                image_id,
                io.BytesIO(image_bytes),
                length=len(image_bytes),
                content_type=content_type
            )
            return image_id
        return await asyncio.to_thread(_upload)

    async def download_image(self, image_id: str) -> bytes:
        def _download():
            response = self.client.get_object(self.bucket, image_id)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        return await asyncio.to_thread(_download)

    async def get_presigned_url(self, image_id: str, expires: int = 3600) -> str:
        def _presigned():
            from datetime import timedelta
            return self.client.presigned_get_object(
                self.bucket,
                image_id,
                expires=timedelta(seconds=expires)
            )
        return await asyncio.to_thread(_presigned)
