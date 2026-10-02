# rag-vllm

vLLM과 PostgreSQL/pgvector를 기본으로 사용하는 문서 수집, 로컬 휴리스틱 품질진단, 근거 검색, 문서 초안 생성 API다. `rag-ollama`를 개인 학습에 쓰고 이 프로젝트를 품질진단·RAG·문서 자동화에 쓰는 구성을 기준으로 한다.

## 지원 범위와 판단 한계

- 텍스트, Markdown, CSV, JSON, HTML, PDF, DOCX, HWPX, PNG/JPEG 계열의 텍스트 추출 및 색인
- UTF-8/CP949 계열 텍스트 정규화, 문서 중복 해시, 청킹, BGE 임베딩, PostgreSQL/pgvector 검색
- 설명 가능한 기본 품질 지표: 빈 문서, 짧은 문서, U+FFFD 대체 문자, 제어 문자, 반복 줄
- 규칙 기반 개인정보 후보 탐지·마스킹과 소규모 내장 동의어 후보 탐지

rag-vllm의 점수는 내부 휴리스틱이다. 공식 공공데이터 품질평가, 개인정보 비식별 확인, 공식 표준사전 일치, 문서 공개 승인으로 해석하면 안 된다. OCR 정확도(CER/WER), 원본과 추출 구조의 보존율, 저작권, 모집단 성공률은 측정하지 않는다. 스캔 PDF에는 OCR을 실행하지 않는다. 이미지 파서는 Tesseract가 설치되고 언어팩이 있을 때에만 OCR을 시도하며, 그 외에는 이미지 정보 안내문을 색인한다.

텍스트 PDF는 `pdfplumber`의 레이아웃 추출과 표 행 추출을 사용한다. 추출 결과는 원본 페이지와 표의 완전한 시각적 보존을 보장하지 않으며 스캔 PDF의 OCR도 수행하지 않는다. HWP 5.0은 현재 `olefile` 기반 로컬 OLE/압축 섹션 파서를 사용한다. `pyhwp`는 오래된 Python 호환 범위와 AGPL 라이선스를 별도 검토하기 전까지 의존성에 추가하지 않았다.

## 저장 및 텍스트 처리

`/documents/file`은 원본 파일 바이트를 별도 보관하지 않고, 추출·정규화된 텍스트를 데이터베이스에 저장한다. 기본값인 `INGEST_AUTO_MASK_PII=true`에서는 휴리스틱 패턴으로 찾은 일부 문자열을 가린 파생 텍스트가 저장된다. 이 설정은 새 등록에 적용되며 기존 문서·청크를 소급해 바꾸지 않는다. 패턴은 모든 개인정보를 찾거나 비식별성을 보장하지 않는다. 이 설정을 `false`로 두면 식별 정보가 포함된 정규화 텍스트가 그대로 저장될 수 있다.

내장 행정 용어 목록은 실험용 후보 사전이다. 기본값인 `INGEST_APPLY_LOCAL_TERM_REPLACEMENTS=false`에서는 후보만 기록하고 원문 용어를 바꾸지 않는다. 이를 `true`로 바꾸면 색인 텍스트에 치환을 적용하지만, 공식 사전 대조나 담당자 승인은 여전히 필요하다. 처리 설정과 후보 수는 문서 메타데이터에 남는다. 입력 본문 또는 원본 바이트는 별도 원문 객체로 보관하지 않는다.

API 업로드 기본 한도는 20 MiB다. PDF는 최대 2,000페이지, DOCX/HWPX 압축 파일은 멤버·압축 해제 크기 제한을 두고, 이미지는 최대 40,000,000픽셀까지 허용한다. 파서 결과와 직접 텍스트 진단은 최대 5,000,000자다. 한도를 넘거나 지원하지 않는 문서는 오류로 반환한다. FastAPI의 multipart 파싱 뒤 라우트 한도가 적용되므로, 네트워크에 노출할 때는 앞단 프록시에도 요청 본문 크기 제한을 설정한다.

`POST /quality/structured`의 `structured-local-v3`는 표본 형식을 추정해 날짜의 달력 유효성과 행 간 컬럼 구성을 살핀다. 행의 값 개수가 헤더와 다르면 스키마 불일치로 집계하고 누락 칸은 결측으로 반영한다. CSV/레코드 입력은 최대 100,000행, 512컬럼, 1,000,000셀로 제한한다. 날짜 형식 추정이 업무별 기준을 대체하지 않으며 결과 점수는 참고용이다.

## 검색 및 집계 라우팅 고도화

다음 기능은 환경변수로 독립 활성화한다. `CONTEXTUAL_RETRIEVAL_ENABLED=true`는 제목, 기존 메타데이터의 `summary`, 청크의 섹션 경로를 **임베딩 입력에만** 붙인다. 원문 청크는 그대로 저장한다. 이미 색인한 문서는 재색인해야 새 문맥 임베딩이 적용된다. 자동 요약 모델 호출은 하지 않는다.

`MULTI_QUERY_ENABLED=true`는 비교·복합 질문을 최대 3개의 근거가 확인된 하위 질의로 계획하고, 저장소 검색을 병렬 실행한 뒤 RRF로 합친다. 답변 생성에는 원래 질문을 사용한다. 하위 계획이 유효하지 않거나 LLM이 없으면 원 질문 검색으로 돌아간다.

`CRAG_ENABLED=true`는 검색 후 질문과 상위 청크의 형태소 근거를 검사한다. 현재 저장소별 점수는 보정된 확률이 아니므로 이 값을 임계치로 간주하지 않는다. 어휘 근거가 부족하면 생성 단계를 건너뛰고 근거 부족을 반환한다. 이는 정답성 증명기가 아니며, 배포 전에 도메인 질의로 거절률과 누락률을 확인한다.

### 분산 시맨틱 캐시

기본 캐시는 프로세스 메모리에서 동작한다. 워커 간 공유와 재시작 후 보존이 필요하면 `uv sync --extra redis-cache`로 Redis 클라이언트를 설치하고 `SEMANTIC_CACHE_REDIS_URL=rediss://...`를 설정한다. Redis 캐시는 보안·프로젝트·문서 범위 및 모델 설정을 포함한 별도 키, TTL, 범위별 항목 제한을 사용한다. 인제스트/삭제 시 캐시 세대를 변경해 이전 답변을 무효화한다. Redis 장애나 클라이언트 미설치는 질의 실패 대신 로컬 메모리 캐시로 폴백한다. Redis에는 생성 답변과 인용 문맥 일부가 저장되므로 내부 접근 제한, TLS, 서버 maxmemory/eviction 정책을 설정한다. `/chat/query`의 앱 전용 질의는 캐시를 사용하지 않는다.

### 구조화 데이터 DuckDB 질의

`query_mode=auto`가 기본값이다. `document_id`가 지정된 CSV/TSV/XLSX에 집계 의도(평균·합계·건수·비율·최소/최대 등)가 분명하면 `/query`와 `/query/stream`은 DuckDB로 자동 라우팅한다. 문서 설명처럼 검색이 필요한 질문은 `query_mode="retrieval"`로 일반 RAG를 선택할 수 있다. `POST /query/structured`는 JSON 레코드, CSV 본문 또는 등록된 표 문서를 직접 지정하는 경로다. XLSX는 문서 파서가 만든 Markdown 표를 사용하며 다중 시트는 `table_index`로 고른다. LLM은 컬럼만 포함한 집계 계획 JSON을 만들고, 서버는 허용된 `count/sum/avg/min/max`, 그룹, 제한된 필터만 SQL로 컴파일한다. 임의 SQL이나 파일 경로는 받지 않는다. 외부 파일 접근은 테이블 적재 후 DuckDB 연결에서 끈다. 이 경로는 CSV/TSV/XLSX 표를 매 요청 처리하며 DuckDB에 업로드 데이터를 영속 저장하지 않는다.

```bash
curl -X POST http://localhost:11020/query/structured \
  -H 'Content-Type: application/json' \
  -d '{"question":"학교별 평균 급식 단가를 계산해줘","records":[{"학교":"한빛초","급식 단가":4500},{"학교":"한빛초","급식 단가":5500},{"학교":"샘물중","급식 단가":6000}]}'
```

앱별 RAG 자격증명은 고정된 `project_name` 안의 표 문서에만 `POST /chat/query/structured`로 질의할 수 있다. 요청은 `question`, `document_id`, 선택적 `table_index`로 제한된다. 앱 프로젝트 범위는 서버 자격증명에서 가져오며 호출자가 바꾸지 못한다.

### 요청별 vLLM LoRA 선택

vLLM에 시작 시 등록한 어댑터만 요청별로 선택한다. `compose.yaml`의 LoRA 인자를 활성화하고 어댑터 경로를 지정한다. 예를 들어 서버를 `--enable-lora --lora-modules moe-standard-v1=/root/lora-adapters/moe-standard-v1`로 실행하고 `.env`에 `LLM_LORA_ADAPTERS_JSON='{"moe-standard-v1":"moe-standard-v1"}'`를 설정한다. `/query` 또는 `/query/stream` 요청에 `"lora_name":"moe-standard-v1"`을 보낸다. rag-vllm은 허용 목록 별칭을 vLLM OpenAI API의 `model` 필드로 매핑한다. 등록되지 않은 별칭은 거부한다. 어댑터의 동적 로드·언로드 API는 노출하지 않는다.

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

RAG-vLLM은 앱 서버가 호출하는 API로 운영한다. `.env.example`은 인증되지 않은 개발 대시보드를 `RAG_DASHBOARD_ENABLED=false`로 비활성화한다. 로컬 운영자가 필요할 때만 신뢰된 개발 환경에서 `RAG_DASHBOARD_ENABLED=true`로 켠다. API는 기본적으로 localhost에서만 듣는다. LAN에서 접근시키려면 `.env`에 `API_HOST=0.0.0.0`과 충분히 강한 `RAG_LAB_API_KEY`를 설정하고, 신뢰된 호출 출처만 방화벽/CORS에 허용한다. API 키는 암호화되지 않은 HTTP에서 평문으로 전송되므로 LAN 접근은 VPN 안에서 사용하거나 TLS 종료 프록시 뒤에 둔다.

### 내부 앱의 챗봇 연결

`POST /chat/query`는 `internal-nas`, `internal-portal`, `food-safety` 같은 앱에서 서버 간 호출로 사용한다. 일반 `/query`와 달리 요청은 `question`, 선택적 `top_k`, 선택적 `generate_answer`만 받으며, 답변 생성은 기본값 `true`다. `generate_answer=false`이면 LLM 답변 생성 없이 해당 범위의 검증된 검색 출처만 반환하므로 MOE 품질 지침 검색에 사용한다. 검색은 요청 본문이 아닌 RAG 서버 자격 설정에서 정한 단일 `project_name` 범위로 제한된다. `X-RAG-Client-ID`는 앱 이름, `X-API-Key`는 그 앱 전용 비밀 키다. 앱별 키는 32자 이상이어야 하며 프런트엔드에 전달하지 않는다. 같은 query 권한 키로 `GET /chat/health`에서 데이터베이스 준비 상태를 확인할 수 있다. 이 경로는 최소한의 `status`와 `database` 상태만 반환한다.

`POST /chat/ingest`는 `ingest` 권한이 있는 앱 자격증명으로 승인된 파생 텍스트를 해당 앱 코퍼스에 넣는다. 본문은 `name`, `text`, 선택적 `replace_existing_source`로 제한하며, 호출자가 보낸 메타데이터나 `project_name`은 받지 않는다. RAG 서버가 코퍼스 범위와 문서 출처 유형을 지정하고 휴리스틱 개인정보 마스킹을 강제한다. 파생본은 앱 서버에서 검토·승인한 뒤 보내며, 원본 경로나 사용자 개인 폴더를 문서 이름으로 쓰지 않는다. 응답은 문서 ID, 청크 수와 품질 점수·등급·개인정보 후보 개수만 반환하고 위치·추천·용어 상세는 제외한다. `POST /chat/delete`는 `delete` 권한이 있는 앱만 같은 앱 코퍼스의 정확히 일치하는 출처 이름을 제거할 수 있어 승인 취소나 파일 삭제 이벤트에 사용할 수 있다. 삭제 후 해당 출처가 없으면 `deleted: true`를 반환하므로 타임아웃 뒤 재시도도 안전하다. 권한 목록을 생략하면 `query`만 허용한다.

같은 앱 프로젝트와 출처 이름에 대한 색인 교체·삭제는 PostgreSQL advisory lock으로 직렬화하며, 잠금 대기는 30초에서 끝나 대기 중인 요청이 API 작업자를 무기한 점유하지 않는다. 외부 Qdrant·Weaviate에서 검색한 청크도 반환 전에 PostgreSQL의 현재 문서 ID와 프로젝트 범위에 대조하므로, 교체·삭제 도중 남은 고아 벡터는 앱 응답에 포함되지 않는다.

서버 환경에 다음과 같은 JSON을 설정한다. 각 앱에는 서로 다른 무작위 키를 만들고 비밀 저장소나 서버 환경변수로 전달한다. 아래의 예시 키는 실제 운영 키로 사용하지 않는다.

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
RAG_CHAT_CLIENTS_JSON='{"internal-nas":{"token":"replace-with-a-generated-nas-token-at-least-32-characters","project_name":"internal-nas","permissions":["query","ingest","delete"]},"internal-portal":{"token":"replace-with-a-generated-portal-token-at-least-32-characters","project_name":"internal-portal","permissions":["query"]},"food-safety":{"token":"replace-with-a-generated-food-token-at-least-32-characters","project_name":"food-safety","permissions":["query","ingest"]},"moe-dq-diag":{"token":"replace-with-a-generated-moe-token-at-least-32-characters","project_name":"moe_dq_diag","permissions":["query"]}}'
```

설명용 자리표시자는 서비스에 적용하기 전에 각각 별도로 생성한 무작위 토큰으로 바꾼다.

각 앱의 서버는 `POST http://<rag-vllm 주소>:11020/chat/query`에 JSON `{"question":"...","top_k":5}`와 `X-RAG-Client-ID`, `X-API-Key` 헤더를 보낸다. 응답은 일반 검색/RAG의 `answer`, `sources`, `confidence_score`, `hallucination_risk`, 가드레일 필드를 포함한다. LMOps 일반 설정이 바뀌어도 이 경로는 질문·답변 원문을 저장하지 않으며 시맨틱 캐시도 사용하지 않는다. 앱 질의는 길이만 남기는 요약 감사 로그를 사용한다. 권한은 앱마다 최소화한다. NAS에는 query/ingest/delete, food-safety에는 query/ingest, internal-portal과 moe-dq-diag에는 query만 부여한다. `project_name`과 허용 작업은 클라이언트가 바꿀 수 없다.

이 앱별 범위는 코퍼스 분리이며 사용자별 파일 권한 검사가 아니다. 해당 프로젝트 이름으로 승인된 공용 자료만 색인해야 한다. 예를 들어 NAS의 사용자 폴더 파일은 파일 ACL을 질의에 연결하기 전까지 색인하지 않고, 승인된 마스킹 파생본만 공용 지식 범위에 넣는다. `moe_dq_diag`는 규칙 설명과 검토 절차 같은 승인 문서를 별도 검색할 수 있지만, RAG 결과는 진단 점수·비교 적격성·공식 판정을 바꾸지 않는 보조 근거로만 표시한다. 브라우저에서 rag-vllm을 직접 호출하거나 앱 키를 브라우저에 제공하지 않는다. TLS 또는 내부 VPN으로 전송 경로를 보호한다.

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

## 프로젝트별 지식 등록

작업공간 루트에서 아래 명령을 실행하면 새 프로젝트의 README, `docs/`, 작업 계약 문서와
주요 빌드·배포 설정을 `project_name` 메타데이터와 함께 등록한다.

```bash
# projects 작업공간 루트에서 실행
./bin/prj rag-sync <project-name>
# 전체 프로젝트:
./bin/prj rag-sync --all
```

등록 문서의 `source_name`은 `project://<project-name>/<relative-path>` 형식이며, 같은
경로의 변경 문서는 이전 색인을 교체한다. `.env`, 소스 코드, 의존성 캐시, 빌드 산출물은
기본 색인 대상이 아니다. 새 프로젝트를 만든 뒤 `./bin/prj workspace`와
`./bin/prj rag-sync <project-name>`을 순서대로 실행한다.

프로젝트 범위를 제한한 검색과 문서 초안은 `project_name`을 함께 보낸다.

```bash
curl -X POST http://localhost:11020/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"이 프로젝트의 실행 방법은?", "project_name":"my-project", "use_llm":true}'
```

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

출력 가드레일 검사를 마친 답변을 SSE 이벤트로 전달하는 질의:

```bash
curl -N -X POST http://localhost:11020/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"연차휴가 발생 요건과 일수를 요약해줘", "top_k": 3, "search_mode": "hybrid"}'
```

Prometheus 서비스 메트릭 수집:

```bash
curl http://localhost:11020/metrics
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

`POST /quality/moe/unstructured/compare`는 `original_file`과 `cleaned_file` 두 업로드를 MOE의 개발 전용 임시 비교 경로에 전달한다. 파일별 최대 크기는 rag-vllm 설정과 25 MiB 중 작은 값이다. 응답은 해시, 파서·규칙 버전, 측정 가능한 차원, 규칙 ID의 해결·추가·잔여 집합을 제공하며, 형식이나 처리 버전이 다르면 이슈 변화는 미측정으로 표시한다. RAG는 파일명과 원문을 보내거나 파일을 저장하지 않으며 공식 품질 승인 판정을 만들지 않는다.

```bash
curl -X POST http://localhost:11020/quality/moe/unstructured/compare \
  -H 'X-API-Key: replace-with-the-value-from-RAG_LAB_API_KEY' \
  -F 'original_file=@./original.pdf' \
  -F 'cleaned_file=@./cleaned.pdf'
```

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

## 고급 엔터프라이즈 기능 (가드레일·평가·LoRA·LMOps·다중 벡터 저장소)

모든 엔드포인트를 로컬로 설정하면 폐쇄망에서도 실행할 수 있다. 기능별 실제 구동 여부는 아래 설정과 산출물 조건을 따른다.

### 1. 보안 및 평가 가드레일 (`src/rag_vllm/guardrails.py`)
- **입력 가드레일 (InputGuardrail)**: 시스템 프롬프트 탈취(System Prompt Leakage), 지침 무시(Instruction Override), DAN/탈옥(Jailbreak), 구분자 오염 공격을 정규식 및 휴리스틱 패턴으로 즉시 탐지하여 차단(`BLOCK`)한다.
- **개인정보 보호**: 주민등록번호, 외국인등록번호, 휴대전화/유선전화, 이메일, 신용카드, 계좌번호를 탐지하여 안전하게 부분 마스킹(`MASK`) 처리한다.
- **출력 가드레일 (OutputGuardrail)**: 답변 내 PII는 마스킹하고 검색 근거성이 크게 부족하거나 시스템 지시문 유출 징후가 있는 답변은 RAG 질의 흐름에서 응답을 보류한다. 중간 위험은 플래그한다. 규칙·단어 겹침 기반 검사다.
- `/draft`와 `/documents/extract`도 입력과 생성 결과를 검사하며, 응답의 `guardrail_action`과 `guardrail_violations`에 판정을 담는다. 이 검사는 패턴 기반 휴리스틱이라 인젝션·PII·환각을 완벽히 탐지하지 않는다.
- **API**: `POST /guardrails/validate-input`, `POST /guardrails/validate-output`

### 2. RAG 정량 평가 프레임워크 (`src/rag_vllm/eval/`)
- **로컬 평가기 (`LocalRagEvaluator`)**: 검색된 실제 문맥과 답변의 단어 겹침을 이용한 휴리스틱 지표다. 의미 이해형 모델 평가는 아니다.
- **DeepEval 어댑터 (`LocalVLLMDeepEvalAdapter`)**: 명시적으로 `--engine deepeval` 또는 API의 `engine=deepeval`을 선택할 때 로컬 vLLM을 판정 모델로 사용한다. 계산 오류는 오류로 반환하며 점수로 대체하지 않는다.
- **실제 RAG 평가 CLI**: 골든셋의 각 `query`를 실행 중인 API에 보내고 검색 문맥·답변·실제 지연시간을 평가한다. `expected_sources`가 있으면 출처 Hit@k·Recall@k·Precision@k를, `expected_guardrail_action`이 있으면 판정 일치율을 계산한다. 답변 평가를 제외할 샘플은 `evaluate_answer: false`로 지정할 수 있다.
- **시연 골든셋**: `data/eval/golden_set.demo.jsonl`은 `scripts/seed_demo.py`의 합성 보안·교육·식품위생 문서용 10개 샘플이다. 실제 업무 지식이나 운영 승인 데이터가 아니므로 운영 배포 기준으로 사용하기 전에 승인된 문서와 담당자 검수 답변으로 교체·확장해야 한다.
- **평가 및 배포 게이트 실행**: rag-vllm API를 실행한 뒤 별도 터미널에서 예시 문서를 적재하고 평가한다. 시연 문서는 로컬 PostgreSQL과 임베딩/LLM 서버가 준비되어야 한다.
  ```bash
  uv run python scripts/seed_demo.py
  ./scripts/run_release_eval.sh
  ```
  게이트 기본 기준은 기대 출처 Hit@k 0.8 이상, 지정 가드레일 판정 일치율 1.0이다. `RAG_EVAL_MIN_SOURCE_HIT_RATE`, `RAG_EVAL_MIN_GUARDRAIL_PASS_RATE` 등 환경 변수로 조정할 수 있고, 결과는 기본 `data/eval/release_report.md`에 저장된다. 이 경로는 Git에서 무시한다. 배포 파이프라인에서도 API와 승인된 평가 문서가 준비된 뒤 스크립트를 호출하면 종료 코드로 게이트를 판정할 수 있다.
- `scripts/run_eval.py` 직접 실행 예: `uv run python scripts/run_eval.py --dataset ./rag_golden_set.jsonl --min-overall-score 70 --min-faithfulness 0.7`. 임계값 미달 시 종료 코드 1을 반환한다.
- 기존 `data/eval_report.md` 수치는 SFT 샘플을 이용한 과거 휴리스틱 결과이며 RAG 벤치마크로 사용하지 않는다.
- **API**: `POST /eval/rag`

### 3. SFT / LoRA 파인튜닝 파이프라인 (`src/rag_vllm/finetuning/`)
- **PEFT LoRA 학습**: `transformers.Trainer`와 `peft.LoraConfig` 기반으로 Qwen3 및 로컬 CausalLM 모델에 대해 LoRA 어댑터(`r=16, alpha=32`)를 로컬 GPU(RTX 5060 Ti 16GB)에서 직접 파인튜닝한다.
- **데이터셋 전처리**: Alpaca 및 Qwen ChatML 포맷을 지원한다. 다중 턴 ChatML은 모든 assistant 턴만 학습하고 system/user 토큰은 loss에서 제외한다.
- **어댑터 병합 & 서빙**: 학습된 어댑터를 베이스 모델과 영구 융합(`merge_and_unload`)하거나 vLLM 동적 LoRA 어댑터 디렉터리로 내보낸다.
- **CLI 도구**: `uv run python scripts/train_lora.py --base-model Qwen/Qwen3-4B-Instruct-2507 --dataset data/sft/alpaca_sft_dataset.json --dry-run`
- 학습은 CLI를 직접 실행해야 한다. 저장소에는 미리 학습된 어댑터가 없으며, Compose의 vLLM LoRA 서빙 옵션도 기본 비활성 상태다.

### 4. 로컬 LMOps & 관측성 (`src/rag_vllm/lmops.py`)
- **생애주기 트레이싱**: 입력 가드레일(`input_guardrail`), 임베딩(`embedding`), 검색(`retrieval`), 리랭킹(`reranking`), LLM 생성(`generation`), 출력 가드레일(`output_guardrail`), 정량 평가(`evaluation`) 단계별 세부 소요 시간(ms)과 소비 토큰을 기록한다.
- **영구 저장 및 모니터링**: PostgreSQL `rag_lmops_traces` 및 `rag_lmops_feedback` 테이블에 저장하며, 로컬 백업용 JSONL 파일(`data/lmops_traces.jsonl`)을 병행 지원한다. 100% 로컬 환경으로 추정 비용($0.00)을 명시한다.
- 기본값은 원 질의·답변·피드백 본문과 사용자 ID를 LMOps에 저장하지 않는다. 정책 승인 후 `LMOPS_STORE_QUERY_TEXT`, `LMOPS_STORE_ANSWER_TEXT`, `LMOPS_STORE_FEEDBACK_TEXT`, `LMOPS_STORE_USER_ID`를 각각 켜면 해당 값을 저장하고 텍스트에는 패턴 기반 PII 마스킹을 적용한다. 피드백 기반 SFT 내보내기에는 질의와 답변 저장을 켜야 한다. 사용자 수정 답변을 반영하려면 피드백 본문 저장도 켠다.
- **SFT 환류 파이프라인**: 사용자 추천(Thumbs Up) 및 평점 4점 이상 또는 수정 답변이 등록된 고품질 QA를 선별하여 파인튜닝 데이터셋(`POST /lmops/export-sft`)으로 내보낸다.
- **API**: `GET /lmops/traces`, `GET /lmops/metrics/summary`, `POST /lmops/feedback`, `POST /lmops/export-sft`

### 5. 다중 벡터 저장소 추상화 인터페이스 (`src/rag_vllm/vector_stores/`)
- **공통 인터페이스 (`BaseVectorStore`)**: PostgreSQL을 문서 메타데이터의 기준 저장소로 유지하고, 선택된 벡터 엔진에도 동일 문서 ID와 청크 임베딩을 기록한다. 검색·스트리밍·삭제는 `VECTOR_STORE_TYPE` 설정을 따른다. pgvector는 FTS+벡터 후보를 RRF로 결합하고, Qdrant와 C++ 엔진은 dense 후보와 Kiwi 키워드 후보를 BM25·RRF로 결합하며, Weaviate는 서버 네이티브 hybrid 검색을 사용한다.
- **지원 엔진**:
  - `PgVectorStore`: 기존 PostgreSQL 16 + pgvector 기반 하이브리드 FTS 검색 연동 (기본값).
  - `QdrantVectorStore`: Qdrant 서버의 영구 컬렉션에 저장한다. 서버 연결 실패 시 메모리 저장소로 조용히 전환하지 않고 요청을 실패시킨다.
  - `WeaviateVectorStore`: 공식 `weaviate-client` v4와 gRPC 배치 업로드·near-vector·hybrid 검색을 사용한다. Weaviate 서버 1.27 이상과 HTTP/gRPC 포트가 모두 필요하다. API key 인증 서버는 `WEAVIATE_API_KEY`를 설정한다.
  - `CppEngineVectorStore`: 사내 C++20 cosine `IndexFlat` 공유 라이브러리를 `ctypes`로 사용하며 문서 메타데이터와 벡터를 프로세스 메모리에 보관한다. 최대 벡터 수 기본값은 25,000개이고, 프로세스 재시작 시 색인이 사라진다. 공유 라이브러리를 빌드해야 한다.
- **검색·출력 보강**: `kiwipiepy` 형태소 토큰과 `rank-bm25`를 한국어 sparse 후보 재정렬에 사용한다. Qdrant는 `search_terms` keyword payload로 dense 검색 밖의 일치 후보도 회수하고 sparse 조회량은 질의당 최대 2,000개로 제한한다. 이 payload는 신규 인제스트 문서부터 채워지므로, 기존 Qdrant 문서는 재인제스트 전까지 형태소 sparse 회수 대상에 포함되지 않는다. `/query/stream`은 완성 문장/길이 창을 출력 가드레일로 검사하고 PII가 감지된 창은 마스킹한 뒤 전송하며, 오류·조기 종료 때도 final `guardrail` SSE 이벤트를 보낸다. `/extract/structured`는 vLLM `response_format` JSON Schema 제약을 요청하고 결과를 Pydantic으로 다시 검증한다. Parent-child 청킹, HyDE, 질의 라우터는 `.env` 플래그로 선택하며 기본 비활성이다.
- **설정 변경**: `.env`의 `VECTOR_STORE_TYPE=pgvector|qdrant|weaviate|cpp_engine`. Qdrant/Weaviate는 서버를 실행하고 URL을 설정한다. C++ 단독 모드는 `CPP_VECTOR_ENGINE_LIBRARY`에 공유 라이브러리 경로를 지정한다. `cpp-vector-engine` 저장소에서 `cmake -S . -B build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF && cmake --build build --target cve_shared`로 빌드할 수 있다. C++ 모드는 앱 문서·청크·감사 trace를 로컬 메모리/JSONL에 저장해 PostgreSQL 없이 기본 수집·검색·삭제·앱 챗봇 API를 제공한다. JSONL은 검색 문서 저장소가 아니며 재시작 뒤 색인은 복구되지 않는다. 데이터 보존이 필요하면 pgvector/Qdrant/Weaviate를 사용한다. 요청별 엔진 변경은 지원하지 않으며 색인에 사용한 설정과 질의 엔진을 일치시킨다. 엔진을 바꾸면 기존 문서를 다시 인제스트한다.
- **API**: `GET /vector-stores/status`
