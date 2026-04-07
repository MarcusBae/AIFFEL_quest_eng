import streamlit as st
import requests
import os

# 페이지 설정
st.set_page_config(
    page_title="AI 로컬 코드 분석 시스템",
    page_icon="📋",
    layout="wide"
)

# 백엔드 API 주소
API_URL = "http://localhost:8000/analyze"

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = st.text_input("🔑 API Key를 입력하세요", type="password", placeholder="test-key-001")
    st.divider()
    st.markdown("""
    ### 💻 지원 언어
    - **Python** (`.py`)
    - **C++** (`.cpp`, `.h`)
    - **Text** (`.txt`)
    """)
    st.info("로컬 Ollama(Llama 3.2) 모델을 사용하여 분석을 수행합니다.")

# 메인 타이틀
st.title("📋 AI 기반 로컬 멀티 언어 코드 분석")
st.markdown("---")

# 1. 파일 업로드 섹션
uploaded_file = st.file_uploader(
    "분석할 소스 코드를 선택하세요", 
    type=["py", "cpp", "c", "h", "hpp", "txt"]
)

if uploaded_file is not None:
    # 파일 내용 읽기
    code_content = uploaded_file.read().decode("utf-8")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 코드 미리보기")
        # 확장자에 따른 언어 하이라이팅 설정
        ext = uploaded_file.name.split('.')[-1].lower()
        lang_map = {'py': 'python', 'cpp': 'cpp', 'c': 'cpp', 'h': 'cpp', 'hpp': 'cpp', 'txt': 'text'}
        
        st.code(code_content, language=lang_map.get(ext, 'python'), line_numbers=True)

    with col2:
        st.subheader("🤖 AI 분석 결과")
        
        # 분석 버튼
        if st.button("🚀 코드 분석 시작", use_container_width=True):
            if not api_key:
                st.warning("먼저 사이드바에서 API Key를 입력해 주세요.")
            else:
                with st.spinner("AI가 코드를 정밀 분석하는 중입니다... (Ollama CPU 추론)"):
                    try:
                        # Multipart 파일 전송 준비
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
                        headers = {"X-API-Key": api_key}
                        
                        # API 호출
                        response = requests.post(API_URL, files=files, headers=headers, timeout=600)
                        
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("success"):
                                st.success(f"✅ 분석 완료! (소요 시간: {data['inference_time']}초)")
                                st.markdown(f"**사용 모델:** `{data['model']}` | **감지된 언어:** `{data['language']}`")
                                st.markdown("---")
                                st.markdown(data["analysis"])
                            else:
                                st.error(f"❌ 분석 실패: {data.get('error', '알 수 없는 오류')}")
                        elif response.status_code == 401:
                            st.error("🔑 인증 실패: 유효하지 않은 API Key입니다.")
                        else:
                            st.error(f"⚠️ 서버 오류: {response.status_code} - {response.text}")
                    
                    except Exception as e:
                        st.error(f"🚨 연결 오류: {str(e)}")
else:
    st.info("파일을 업로드하면 코드 미리보기와 AI 분석을 시작할 수 있습니다.")

# 푸터
st.markdown("---")
st.caption("© 2026 AI Code Analyzer - Built with FastAPI, Streamlit and Ollama")
