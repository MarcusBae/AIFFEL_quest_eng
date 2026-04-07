import httpx
from typing import Dict, Any

# Ollama 로컬 서버 정보
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"

async def generate_analysis(filename: str, code_content: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Ollama API를 통해 코드 분석 생성"""
    
    # 확장자에 따른 언어 판별
    ext = filename.split('.')[-1].lower()
    language = "Python" if ext in ['py', 'pyw'] else "C++" if ext in ['cpp', 'c', 'h', 'hpp'] else "Unknown"
    
    # 언어별 맞춤형 프롬프트 생성
    prompt = f"""
    당신은 세계적인 소프트웨어 아키텍트이자 코드 분석 전문가입니다. 
    아래 제출된 {language} 소스 코드를 분석하고 리포트를 작성해 주세요.
    
    [파일명]: {filename}
    [언어]: {language}
    
    [소스 코드]:
    {code_content[:10000]} # 토큰 제한 방지 (최대 10k자)
    
    분석 양식:
    1. **주요 기능 요약**: 코드가 수행하는 핵심 역할을 한 문장으로 설명.
    2. **잠재적 버그/위험**: 로직 결함, 보안 취약점 또는 스타일 위협 요소.
    3. **제안 사항**: {language} 특성에 맞는 개선 제안 (예: 메모리 관리, PEP8 등).
    
    반드시 마크다운 형식을 사용하여 답변해 주세요.
    """
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2048,
            "num_ctx": 32768
        }
    }
    
    async with httpx.AsyncClient(timeout=1200.0) as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            return {
                "language": language,
                "analysis": result.get("response", "분석 결과를 생성하지 못했습니다."),
                "model": model_name,
                "success": True
            }
        except Exception as e:
            return {
                "language": language,
                "analysis": f"추론 도중 오류 발생: {str(e)}",
                "model": model_name,
                "success": False,
                "error": str(e)
            }
