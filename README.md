# rag-vllm

vLLM과 PostgreSQL/pgvector를 사용하는 문서 수집, 로컬 휴리스틱 품질진단, 근거 검색, 문서 초안 생성 API다. `rag-ollama`를 개인 학습에 쓰고 이 프로젝트를 품질진단·RAG·문서 자동화에 쓰는 구성을 기준으로 한다.

## 지원 범위와 판단 한계

- 텍스트, Markdown, CSV, JSON, HTML, PDF, DOCX, HWPX, PNG/JPEG 계열의 텍스트 추출 및 색인
- UTF-8/CP949 계열 텍스트 정규화, 문서 중복 해시, 청킹, BGE 임베딩, PostgreSQL/pgvector 검색
- 설명 가능한 기본 품질 지표: 빈 문서, 짧은 문서, U+FFFD 대체 문자, 제어 문자, 반복 줄
- 규칙 기반 개인정보 후보 탐지·마스킹과 소규모 내장 동의어 후보 탐지

rag-vllm의 점수는 내부 휴리스틱이다. 공식 공공데이터 품질평가, 개인정보 비식별 확인, 공식 표준사전 일치, 문서 공개 승인으로 해석하면 안 된다. OCR 정확도(CER/WER), 원본과 추출 구조의 보존율, 저작권, 모집단 성공률은 측정하지 않는다. 스캔 PDF에는 OCR을 실행하지 않는다. 이미지 파서는 Tesseract가 설치되고 언어팩이 있을 때에만 OCR을 시도하며, 그 외에는 이미지 정보 안내문을 색인한다.

## 저장 및 텍스트 처리

`/documents/file`은 원본 파일 바이트를 별도 보관하지 않고, 추출·정규화된 텍스트를 데이터베이스에 저장한다. 기본값인 `INGEST_AUTO_MASK_PII=true`에서는 휴리스틱 패턴으로 찾은 일부 문자열을 가린 파생 텍스트가 저장된다. 이 설정은 새 등록에 적용되며 기존 문서·청크를 소급해 바꾸지 않는다. 패턴은 모든 개인정보를 찾거나 비식별성을 보장하지 않는다. 이 설정을 `false`로 두면 식별 정보가 포함된 정규화 텍스트가 그대로 저장될 수 있다.

내장 행정 용어 목록은 실험용 후보 사전이다. 기본값인 `INGEST_APPLY_LOCAL_TERM_REPLACEMENTS=false`에서는 후보만 기록하고 원문 용어를 바꾸지 않는다. 이를 `true`로 바꾸면 색인 텍스트에 치환을 적용하지만, 공식 사전 대조나 담당자 승인은 여전히 필요하다. 처리 설정과 후보 수는 문서 메타데이터에 남는다. 입력 본문 또는 원본 바이트는 별도 원문 객체로 보관하지 않는다.

API 업로드 기본 한도는 20 MiB다. PDF는 최대 2,000페이지, DOCX/HWPX 압축 파일은 멤버·압축 해제 크기 제한을 두고, 이미지는 최대 40,000,000픽셀까지 허용한다. 파서 결과와 직접 텍스트 진단은 최대 5,000,000자다. 한도를 넘거나 지원하지 않는 문서는 오류로 반환한다. FastAPI의 multipart 파싱 뒤 라우트 한도가 적용되므로, 네트워크에 노출할 때는 앞단 프록시에도 요청 본문 크기 제한을 설정한다.

`POST /quality/structured`의 `structured-local-v3`는 표본 형식을 추정해 날짜의 달력 유효성과 행 간 컬럼 구성을 살핀다. 행의 값 개수가 헤더와 다르면 스키마 불일치로 집계하고 누락 칸은 결측으로 반영한다. CSV/레코드 입력은 최대 100,000행, 512컬럼, 1,000,000셀로 제한한다. 날짜 형식 추정이 업무별 기준을 대체하지 않으며 결과 점수는 참고용이다.

## GPU 및 모델 설정

기본 vLLM 모델은 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)이다. 모델 카드의 최대 컨텍스트는 262K 토큰이지만, 이 프로젝트는 16 GiB GPU에서 VRAM 사용량을 제한하도록 `VLLM_MAX_MODEL_LEN=8192`, `VLLM_GPU_MEMORY_UTILIZATION=0.75`를 기본으로 둔다. BGE 임베딩은 GPU 여유를 위해 기본 CPU 설정을 사용한다. 실제 처리량과 동시 요청 수는 GPU와 입력 길이에 따라 조정해야 한다.

Compose 이미지 기본값은 재현성을 위해 공식 vLLM `v0.30.0`으로 고정했다. 이 이미지의 기본 CUDA 13 계열은 호스트 NVIDIA R580 이상 드라이버가 필요하며, 현재 개발 장비 드라이버는 `616.92`다. [vLLM 릴리스](https://github.com/vllm-project/vllm/releases), [GPU 설치/드라이버 호환 안내](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

공유 개발 장비의 RTX 5060 Ti 16 GiB에서는 `gpu-memory-utilization=0.85`로 vLLM이 약 14.7 GiB를 사용한 상태가 관찰됐다. `rag-ollama`의 `gpt-oss:20b`와 vLLM을 동시에 GPU에 올리지 않는다. 모델 서버를 전환할 때 기존 서버를 먼저 중지한다.

```bash
cp -n .env.example .env
uv sync
# Ollama의 gpt-oss:20b가 GPU 메모리에 올라와 있다면 먼저 내린다:
# ollama stop gpt-oss:20b
docker compose --profile vllm up -d
uv run rag-vllm-api
```

기본 포트는 vLLM `11435`, rag-vllm API `11020`이다. API는 호스트에서 실행하므로 `.env`의 `LLM_BASE_URL=http://127.0.0.1:11435/v1` 설정으로 vLLM 컨테이너에 연결한다.

API는 기본적으로 localhost에서만 듣는다. LAN에서 접근시키려면 `.env`에 `API_HOST=0.0.0.0`과 충분히 강한 `RAG_LAB_API_KEY`를 설정하고, 신뢰된 호출 출처만 방화벽/CORS에 허용한다. API 키는 암호화되지 않은 HTTP에서 평문으로 전송되므로 LAN 접근은 VPN 안에서 사용하거나 TLS 종료 프록시 뒤에 둔다. 대시보드 페이지에서 키를 입력하면 현재 페이지 메모리에만 보관하고 API 요청에 `X-API-Key`를 붙인다. 페이지를 새로고침하면 다시 입력해야 한다. 서버 API와 `/health`는 키를 확인하며, HTML 대시보드 페이지는 키 입력을 위해 공개 제공된다. 외부 네트워크에 인증 없이 노출하지 않는다.

`department`와 `max_security_level` 질의 값은 검색 범위를 좁히는 호출자 지정 필터다. 이 API 키는 사용자 신원이나 문서별 권한을 구분하지 않으므로, 해당 필터를 접근통제로 간주하지 않는다. 민감 문서를 여러 사용자에게 제공하려면 인증된 서버 주체에서 권한을 계산하고 DB 질의에 적용하는 구성이 추가되어야 한다.

```bash
curl http://localhost:11020/health \
  -H 'X-API-Key: replace-with-the-value-from-RAG_LAB_API_KEY'
```

```bash
curl http://localhost:11435/health
curl http://localhost:11435/v1/models \
  -H 'Authorization: Bearer vllm-local'
```

아래 rag-vllm API 예시는 `RAG_LAB_API_KEY`를 설정하지 않은 기본 localhost 구성 기준이다. 키를 설정한 경우 각 rag-vllm API 요청에 `-H 'X-API-Key: <실제 키>'`를 추가한다.

vLLM을 내리고 Ollama 모델을 사용하는 경우:

```bash
docker compose --profile vllm stop vllm
```

`EMBEDDING_PROVIDER=hash`는 연결 확인용 결정론적 임베딩으로 의미 검색에 적합하지 않다. 실제 검색에는 기본 `EMBEDDING_PROVIDER=flag`, `BAAI/bge-m3`를 사용한다. Hugging Face에서 모델을 내려받을 수 있어야 한다.

## API 예시

텍스트 등록:

```bash
curl -X POST http://localhost:11020/documents/text \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "휴가규정.md",
    "text": "연차휴가는 1년간 80퍼센트 이상 출근한 근로자에게 부여한다.",
    "metadata": {"department": "인사", "version": "2026-01"}
  }'
```

파일 등록:

```bash
curl -X POST http://localhost:11020/documents/file \
  -F 'file=@./sample.pdf' \
  -F 'metadata={"department":"품질"}'
```

검색/RAG:

```bash
curl -X POST http://localhost:11020/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"연차휴가 조건은 무엇인가?", "top_k": 5, "min_quality_score": 80, "use_llm": true}'
```

LLM 설정이 없거나 관련 문서가 없으면 `answer`는 `null` 또는 설명 문자열일 수 있다. 응답의 `sources`를 검토하고, 생성된 초안은 근거 문서와 대조한다.

휴리스틱 품질 및 용어 후보 확인:

```bash
curl -X POST http://localhost:11020/quality/enterprise \
  -H 'Content-Type: application/json' \
  -d '{"text":"담당자 연락처는 010-1234-5678이며 과태금 기준을 확인한다.", "auto_mask_pii": true, "apply_local_term_replacements": false}'
```

`standardization_status`, `standardization_source`, `standardization_applied` 응답 필드는 후보의 출처와 적용 여부를 구분한다. 용어 후보는 승인 전에 자동 적용하지 않는다.

`POST /quality/diagnose`에 문서 ID를 주면 현재 `baseline-text-v1` 규칙으로 저장된 추출 텍스트를 다시 진단한다. 이 요청은 문서 등록 시 저장한 기존 리포트를 덮어쓰거나 별도 이력으로 저장하지 않는다. `/documents/{id}`에는 등록 시점 리포트가 남고, 재진단 응답에는 실행한 규칙 버전이 표시된다. 업무 프로파일과 규칙 버전별 이력 테이블은 업무 기준·검증 사례를 확정한 뒤 추가할 리팩터링 후보다.

검색 감사 로그는 질의 원문 대신 `omitted;chars:<length>`만 저장한다. 이는 비식별 처리가 아니라 원문을 저장하지 않는 정책이다. 이전 버전에서 만들어진 원문 질의 로그는 DB에서 자동 삭제하지 않으며, API 조회에서는 `legacy_query_redacted`로 가려서 반환한다.

## `moe_dq_diag`와의 경계

`moe_dq_diag`의 개발 API는 문서 1건 임시 파싱을 위해 `POST /api/v1/quality/documents/diagnose`를 제공한다. 이 API는 개발 환경에서만 동작하고, 업로드 한도는 25 MiB이며, 파일을 임시 처리 후 삭제하고 원문이 섞일 수 있는 issue sample은 응답에서 제거한다. 응답의 `unstructured-quality-v1`은 추출 가능성만 측정하고 `release_eligible=false`를 유지한다. CER/WER, OCR 정확도, 구조 보존율, 개인정보 마스킹, 저작권, 운영 적합성을 입증하지 않는다.

`GET /api/v1/standardization/term-matches?table=...`는 설치된 사전으로 기존 C-2 테이블의 컬럼명을 읽기 전용 대조한다. 임의 비정형 문서의 본문 용어를 받거나 매핑을 승인하는 API가 아니며, 응답은 도메인 검토 대기 상태다 (`sourceRowsRead=0`, `databaseWrites=0`). rag-vllm의 내장 후보 목록과 이 MOE 표준사전 후보는 서로 대체할 수 없다.

따라서 rag-vllm의 점수나 동의어 후보를 MOE 표준화·공개 판정의 증거로 사용하지 않는다. 비정형 문서의 정식 표준화 검증에는 승인된 사전·문서 필드 매핑·도메인 검토 계약이 추가로 필요하다.

rag-vllm의 기본 `MOE_DQ_API_BASE_URL`은 `http://127.0.0.1:8001`이다. 이는 MOE Compose의 호스트 포트이며 FastAPI 직접 실행 포트는 `http://127.0.0.1:8000`이다. MOE API가 꺼져 있거나 연결할 수 없으면 `503`, 응답 시간이 초과되면 `504`를 반환한다. 진단 응답은 5 MiB까지 받고, 예상한 개발 진단 계약을 벗어난 응답은 거부한다. 연동을 끄려면 `.env` 값을 비운다.

MOE 개발 API를 실행하려면 프로젝트 폴더에서 Compose 서비스를 시작한다.

```bash
(cd ../moe_dq_diag && docker compose up -d moe-dq-api)
```

```bash
# 기본값: MOE_DQ_API_BASE_URL=http://127.0.0.1:8001
curl -X POST http://localhost:11020/quality/moe/unstructured \
  -H 'X-API-Key: replace-with-the-value-from-RAG_LAB_API_KEY' \
  -F 'file=@./sample.pdf'

curl --get http://localhost:11020/quality/moe/c2-term-matches \
  -H 'X-API-Key: replace-with-the-value-from-RAG_LAB_API_KEY' \
  --data-urlencode 'table=your_c2_table'
```

업로드는 rag-vllm의 `MAX_UPLOAD_BYTES`와 MOE 연동 25 MiB 중 더 작은 한도까지만 전달하고, rag-vllm DB에는 저장하지 않는다. 원본 파일명 대신 확장자만 전달하며, 응답에서 원문 샘플 필드를 제거한다. MOE 개발 API도 임시 파싱 후 파일을 지운다. `/quality/moe/unstructured`는 같은 업로드를 MOE의 임시 진단 API와 rag-vllm의 로컬 텍스트 기준 진단기에 각각 전달한다. 결과는 MOE `report`와 `local_comparison.rag_vllm` 아래에 분리해 반환하며, 두 점수는 측정 범위가 달라 비교·합산하지 않는다. rag-vllm 파서가 해당 파일을 읽지 못하면 MOE 결과를 유지하고 로컬 비교만 `available=false`로 표시한다. 어느 결과도 rag-vllm DB에 저장하지 않는다. 두 번째 경로는 기존 C-2 테이블 컬럼명 후보만 읽는다. 문서 본문 임의 용어의 승인된 표준화 검증 기능은 아직 없으며, 두 결과를 하나의 공식 판정으로 사용하지 않는다. `RAG_LAB_API_KEY`를 설정했다면 예시 헤더의 값을 실제 키로 바꾸고, 키를 설정하지 않았다면 헤더 두 줄을 생략한다.

C-2 프록시 결과에는 표준사전 `dictionaryVersions`, 검토 상태 `decisionStatus`, 원천 행 읽기 수, DB 쓰기 수가 포함된다. 기대한 버전·검토·읽기 전용 계약이 빠지면 응답을 거부한다.

로컬 개발 환경에서 직접 호출할 때는 `moe_dq_diag` Compose의 호스트 포트 `8001`을 사용한다 (`8001` → 컨테이너 `8000`). FastAPI를 직접 실행했다면 `8000`을 사용한다. 아래 명령은 파일을 개발 전용 임시 진단 경로에 보내며 저장·승인하지 않는다.

```bash
# Compose: localhost:8001. Direct FastAPI: localhost:8000으로 바꿔 실행.
curl -X POST http://localhost:8001/api/v1/quality/documents/diagnose \
  -F 'file=@./sample.pdf'

# Read-only C-2 column-name candidates; provide an existing table name.
curl --get http://localhost:8001/api/v1/standardization/term-matches \
  --data-urlencode 'table=your_c2_table'
```

두 응답 모두 개발/검토용 근거다. 문서 진단은 `release_eligible=false`이며 OCR·정확도·PII·저작권 또는 운영 적격을 확정하지 않는다. `term-matches`는 C-2 DB의 컬럼명만 읽으며 비정형 문서 본문을 표준화하지 않고, 후보 승인을 기록하지 않는다.
