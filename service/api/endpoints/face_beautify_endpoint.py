from fastapi import APIRouter
router = APIRouter()

from service.api.schemas.face_beautify_schema import FaceBeautifyRequest, FaceBeautifyResponse
from beautify_pipeline.pipeline import pipeline


@router.post("/face_beautify/")
async def face_beautify(request: FaceBeautifyRequest) -> FaceBeautifyResponse:
    """
    Beautify the face in the image

    Args:
    - image: base64 encoded image

    Returns:
    - image: base64 encoded image
    """
    beautified_image = pipeline(request.image)
    return FaceBeautifyResponse(image=beautified_image)

