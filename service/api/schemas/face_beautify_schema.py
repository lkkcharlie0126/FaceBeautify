import base64
from pydantic import BaseModel, validator
import binascii


class FaceBeautifyRequest(BaseModel):
    image: str # base64 encoded image

    @validator('image')
    def validate_image(cls, v):
        try:
            base64.b64decode(v)
        except binascii.Error:
            raise ValueError('Invalid base64 image')
        return v


class FaceBeautifyResponse(BaseModel):
    image: str # base64 encoded image