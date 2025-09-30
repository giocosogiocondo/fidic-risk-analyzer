# FIDIC Risk Analyzer

계약서(텍스트/ PDF/ DOCX)를 FIDIC 분쟁 사례(JSON)로부터 학습한 임베딩 인덱스와 비교해, 위험(분쟁 소지) 태그별로 스코어링하고 유사 사례 근거를 함께 제시하는 분석 도구입니다.  
FAISS(선택)와 CrossEncoder 리랭크(선택)를 지원하며, 임베딩 캐시로 반복 실행 속도를 개선합니다.  
CLI 한 줄로 JSON/Markdown 리포트를 생성할 수 있습니다. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

## 주요 기능

- **계약서 파싱 및 조항 분리**: `.txt/.pdf/.docx`에서 텍스트 추출, FIDIC 스타일의 조항 번호(예: `13.8`, `20.1`)로 블록화. :contentReference[oaicite:3]{index=3}
- **케이스 조각 인덱스 생성**: `core_issues / risk_lessons / factual_triggers / errors_identified / snippets` 섹션을 통합 인덱스로 구축. :contentReference[oaicite:4]{index=4}
- **임베딩 백엔드 자동 선택**: `sentence-transformers` 사용 가능 시 E5 등 모델 사용, 미설치 시 TF-IDF로 폴백. :contentReference[oaicite:5]{index=5}
- **임베딩 캐시**: 사례 인덱스 임베딩을 `.npz`로 저장/재사용(모델/아이템 수 자동 검증). :contentReference[oaicite:6]{index=6}
- **고속 검색(옵션)**: FAISS(Inner Product ≈ Cosine)로 1차 후보 검색. :contentReference[oaicite:7]{index=7}
- **정확도 향상(옵션)**: CrossEncoder(`ms-marco-MiniLM-L-6-v2`)로 top-50 → top-10 리랭크. :contentReference[oaicite:8]{index=8}
- **태그별 스코어링**: 키워드/클로즈 매칭 보정, 섹션 가중치, 상위 유사도 통계로 Confidence 산출. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
- **리포트 출력**: JSON/Markdown 결과물 생성(태그 모드 & 검색 전용 모드). :contentReference[oaicite:11]{index=11}

---

## 설치

```bash
# (옵션) 가상환경
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필수
pip install -r requirements.txt  # 문서에 맞게 구성해 주세요

# 선택적 가속/정확도 향상
pip install faiss-cpu sentence-transformers
pip install pdfminer.six python-docx  # PDF/DOCX 지원
