from typing import List, Optional

from pydantic import BaseModel


class AlertVerificationRequest(BaseModel):
    alertInstanceId: int
    eventTypeId: int
    cameraId: str
    filterPrompt: Optional[str] = None
    images: Optional[List[str]] = None
    requestStartTimestamp: Optional[int] = None


class AlertVerificationResponse(BaseModel):
    alertInstanceId: int
    cameraId: str
    verified: bool
    success: bool
    message: Optional[str] = None
