"""
Day 6 - API Key 인증
"""
from fastapi import HTTPException, Header

# API Key 설정
# 실무에서는 환경 변수나 시크릿 매니저에서 로드합니다.
VALID_API_KEYS = {
    "test-key-001": "사용자A",
    "test-key-002": "사용자B",
}

# *your code* — Header에서 키 추출
async def verify_api_key(x_api_key: str = Header(None)) -> str:  
    """
    API Key를 검증합니다.
    FastAPI의 Depends()로 엔드포인트에 주입합니다.

    Returns:
        인증된 사용자 이름
    Raises:
        HTTPException: 키가 없거나 유효하지 않을 때
    """
    if x_api_key is None:
        # *your code* — Unauthorized
        raise HTTPException(
            status_code=401,                       
            detail="API Key가 필요합니다. X-API-Key 헤더를 포함해 주세요.",
        )

    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 API Key입니다.",
        )

    return VALID_API_KEYS[x_api_key]