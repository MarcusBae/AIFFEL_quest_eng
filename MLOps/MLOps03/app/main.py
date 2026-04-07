import asyncio
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from app.auth import verify_api_key
from app.schemas import AnalysisResponse
from app.model_service import generate_analysis
from concurrent.futures import ThreadPoolExecutor
import time

app = FastAPI(title="AI 로컬 코드 분석 시스템")

# PRD 요구사항: 비동기 처리를 위한 ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/")
def read_root():
    return {"message": "AI Multi-language Code Analysis API is running!"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_code(
    file: UploadFile = File(...),
    username: str = Depends(verify_api_key)
):
    """업로드된 소스 코드 파일을 AI 모델(Ollama)이 분석"""
    
    # 1. 파일 확장자 검증
    filename = file.filename
    ext = filename.split('.')[-1].lower()
    allowed_exts = ['py', 'cpp', 'c', 'h', 'hpp', 'txt']
    
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=422,
            detail=f"허용되지 않은 파일 형식입니다. (허용: {', '.join(allowed_exts)})"
        )
    
    # 2. 파일 내용 디코딩
    try:
        content = await file.read()
        code_content = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"파일을 읽는 중 오류가 발생했습니다: {str(e)}"
        )
    
    # 3. 비동기 추론 요청 (model_service 호출)
    start_time = time.time()
    analysis_result = await generate_analysis(filename, code_content)
    end_time = time.time()
    inference_time = round(end_time - start_time, 2)
    
    # 4. 결과 반환
    return AnalysisResponse(
        filename=filename,
        language=analysis_result["language"],
        model=analysis_result["model"],
        analysis=analysis_result["analysis"],
        inference_time=inference_time,
        success=analysis_result["success"],
        error=analysis_result.get("error")
    )
