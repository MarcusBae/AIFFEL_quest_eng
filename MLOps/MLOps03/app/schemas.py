from pydantic import BaseModel, Field
from typing import Optional

class AnalysisResponse(BaseModel):
    """코드 분석 결과 응답 스키마"""
    filename: str = Field(..., description="분석된 파일명")
    language: str = Field(..., description="코드 언어 (python, cpp 등)")
    model: str = Field(..., description="분석에 사용된 AI 모델명")
    analysis: str = Field(..., description="AI가 생성한 분석 결과 리포트 (Markdown)")
    inference_time: float = Field(0.0, description="추론 소요 시간 (초)")
    success: bool = Field(True, description="요청 처리 성공 여부")
    error: Optional[str] = Field(None, description="에러 발생 시 상세 메시지")
