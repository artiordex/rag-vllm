# =============================================================================
# 파일명: lmops.py
# 경로: src/rag_vllm/lmops.py
# 목적: RAG 생애주기 추적, 단계별 스팬 계측, 피드백 수집 및 SFT 데이터 환류 제공함
# 작성자: AI전략팀
# 작성일: 2026-10-01
# 수정일: 2026-10-01
# =============================================================================

"""RAG 생애주기 추적, 단계별 스팬 계측, 피드백 수집 및 SFT 데이터 환류 제공함"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import Settings
from .standardization import detect_and_mask_pii

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TraceSpan:
    """개별 RAG 하위 단계(임베딩, 검색, 생성, 가드레일 등)의 실행 시간과 메타데이터임"""

    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, metadata: dict[str, Any] | None = None) -> None:
        """스팬을 종료하고 소요 시간을 밀리초 단위로 계산함"""
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        """스팬 정보를 딕셔너리로 직렬화함"""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TraceContext:
    """단일 질의응답 요청 전체의 생애주기 트레이스를 수집 및 관리함"""

    trace_id: UUID = field(default_factory=uuid4)
    client_ip: str | None = None
    query_text: str = ""
    answer_text: str | None = None
    model_name: str = "rag-vllm-model"
    vector_store_type: str = "pgvector"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    guardrail_action: str = "allow"
    guardrail_violations: list[dict[str, Any]] = field(default_factory=list)
    faithfulness_score: float | None = None
    answer_relevance_score: float | None = None
    hallucination_risk: str | None = None
    start_time: float = field(default_factory=time.perf_counter)
    total_latency_ms: float = 0.0
    spans: list[TraceSpan] = field(default_factory=list)

    def start_span(self, name: str, metadata: dict[str, Any] | None = None) -> TraceSpan:
        """새로운 하위 단계 스팬을 시작하고 컨텍스트에 등록함"""
        span = TraceSpan(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )
        self.spans.append(span)
        return span

    def finish(self) -> None:
        """전체 트레이스를 종료하고 총 소요 시간을 기록함"""
        elapsed = time.perf_counter() - self.start_time
        self.total_latency_ms = round(elapsed * 1000.0, 2)

    def to_dict(self) -> dict[str, Any]:
        """트레이스 전체 레코드를 직렬화함"""
        return {
            "trace_id": str(self.trace_id),
            "client_ip": self.client_ip,
            "query_text": self.query_text,
            "answer_text": self.answer_text,
            "model_name": self.model_name,
            "vector_store_type": self.vector_store_type,
            "total_latency_ms": self.total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": 0.0,  # 100% 로컬 인프라로 추가 비용 0원임
            "guardrail_action": self.guardrail_action,
            "guardrail_violations": self.guardrail_violations,
            "faithfulness_score": self.faithfulness_score,
            "answer_relevance_score": self.answer_relevance_score,
            "hallucination_risk": self.hallucination_risk,
            "spans": [s.to_dict() for s in self.spans],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# LMOps 영구 저장 및 비동기/버퍼링 매니저
# =============================================================================

class LmopsManager:
    """PostgreSQL 및 로컬 JSONL 파일에 트레이스와 사용자 피드백을 기록함"""

    def __init__(self, settings: Settings, log_file: str | None = None) -> None:
        self.settings = settings
        self.log_file = Path(log_file or settings.lmops_log_file)
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """로컬 로그 디렉터리와 파일을 준비함"""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def init_schema(self, conn: psycopg.Connection[Any]) -> None:
        """LMOps 트레이스 및 피드백 테이블 스키마를 초기화함"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_lmops_traces (
                id UUID PRIMARY KEY,
                client_ip TEXT,
                query_text TEXT NOT NULL,
                answer_text TEXT,
                model_name TEXT,
                vector_store_type TEXT,
                total_latency_ms REAL NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                guardrail_action TEXT,
                guardrail_violations JSONB DEFAULT '[]'::jsonb,
                faithfulness_score REAL,
                answer_relevance_score REAL,
                hallucination_risk TEXT,
                spans JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_lmops_traces_created_at
            ON rag_lmops_traces (created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_lmops_feedback (
                id BIGSERIAL PRIMARY KEY,
                trace_id UUID NOT NULL REFERENCES rag_lmops_traces(id) ON DELETE CASCADE,
                thumbs INTEGER,
                rating INTEGER,
                comment TEXT,
                corrected_answer TEXT,
                user_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_lmops_feedback_trace_id
            ON rag_lmops_feedback (trace_id)
            """
        )

    def record_trace(
        self,
        trace: TraceContext,
        *,
        store_query_text: bool | None = None,
        store_answer_text: bool | None = None,
    ) -> None:
        """완료된 트레이스를 데이터베이스 및 로컬 JSONL 파일에 기록함"""
        trace_data = trace.to_dict()
        # Guardrail detections can carry the exact matched phrase and its offsets.
        # The category and rule are enough for aggregate review; do not retain
        # request substrings through this side channel when query text is disabled.
        violations = trace_data.get("guardrail_violations")
        if isinstance(violations, list):
            safe_fields = {"rule_name", "category", "severity", "message"}
            trace_data["guardrail_violations"] = [
                {key: value for key, value in item.items() if key in safe_fields}
                for item in violations
                if isinstance(item, dict)
            ]
        query_text_enabled = (
            self.settings.lmops_store_query_text if store_query_text is None else store_query_text
        )
        answer_text_enabled = (
            self.settings.lmops_store_answer_text if store_answer_text is None else store_answer_text
        )

        # TraceContext는 요청 원문을 포함할 수 있으므로 저장 경계에서 다시 필터링함.
        # 보관이 명시적으로 허용된 경우에도 패턴 기반 PII 마스킹을 항상 적용함.
        if query_text_enabled:
            safe_query, _ = detect_and_mask_pii(trace.query_text, mask=True)
            trace_data["query_text"] = safe_query
        else:
            trace_data["query_text"] = ""

        if answer_text_enabled and trace.answer_text:
            safe_answer, _ = detect_and_mask_pii(trace.answer_text, mask=True)
            trace_data["answer_text"] = safe_answer
        else:
            trace_data["answer_text"] = None

        # 1. 로컬 JSONL 파일 백업 기록함
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("로컬 LMOps JSONL 파일 기록 실패: %s", exc)

        # C++ 단독 모드에서는 JSONL을 감사 추적의 로컬 저장소로 사용하며 DB 연결을 시도하지 않음.
        if self.settings.vector_store_type == "cpp_engine":
            return

        # 2. PostgreSQL 저장함
        from .db import _connect

        try:
            with _connect(self.settings) as conn:
                self.init_schema(conn)
                conn.execute(
                    """
                    INSERT INTO rag_lmops_traces (
                        id, client_ip, query_text, answer_text, model_name,
                        vector_store_type, total_latency_ms, prompt_tokens,
                        completion_tokens, total_tokens, guardrail_action,
                        guardrail_violations, faithfulness_score,
                        answer_relevance_score, hallucination_risk, spans, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()
                    )
                    """,
                    (
                        trace.trace_id,
                        trace.client_ip,
                        trace_data["query_text"],
                        trace_data["answer_text"],
                        trace.model_name,
                        trace.vector_store_type,
                        trace.total_latency_ms,
                        trace.prompt_tokens,
                        trace.completion_tokens,
                        trace.total_tokens,
                        trace.guardrail_action,
                        Jsonb(trace_data["guardrail_violations"]),
                        trace.faithfulness_score,
                        trace.answer_relevance_score,
                        trace.hallucination_risk,
                        Jsonb([s.to_dict() for s in trace.spans]),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("PostgreSQL LMOps 트레이스 저장 실패: %s", exc)

    def record_feedback(
        self,
        *,
        trace_id: UUID,
        thumbs: int | None = None,
        rating: int | None = None,
        comment: str | None = None,
        corrected_answer: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """사용자 피드백을 기록하고 피드백 ID를 반환함"""
        from .db import _connect

        if getattr(self.settings, "lmops_store_feedback_text", False):
            safe_comment, _ = detect_and_mask_pii(comment or "", mask=True)
            safe_corrected_answer, _ = detect_and_mask_pii(corrected_answer or "", mask=True)
            stored_comment = safe_comment or None
            stored_corrected_answer = safe_corrected_answer or None
        else:
            stored_comment = None
            stored_corrected_answer = None
        stored_user_id = user_id if getattr(self.settings, "lmops_store_user_id", False) else None

        with _connect(self.settings) as conn:
            self.init_schema(conn)
            cur = conn.execute(
                """
                INSERT INTO rag_lmops_feedback (
                    trace_id, thumbs, rating, comment, corrected_answer, user_id
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (trace_id, thumbs, rating, stored_comment, stored_corrected_answer, stored_user_id),
            )
            row = cur.fetchone()
            conn.commit()
            return int(row["id"]) if row else 0

    def get_recent_traces(self, limit: int = 50, action_filter: str | None = None) -> list[dict[str, Any]]:
        """최근 트레이스 목록을 조회함"""
        from .db import _connect

        try:
            with _connect(self.settings) as conn:
                self.init_schema(conn)
                query = "SELECT * FROM rag_lmops_traces"
                params: list[Any] = []
                if action_filter:
                    query += " WHERE guardrail_action = %s"
                    params.append(action_filter)
                query += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)

                cur = conn.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for r in rows:
                    item = dict(r)
                    item["id"] = str(item["id"])
                    item["created_at"] = item["created_at"].isoformat() if item.get("created_at") else None
                    results.append(item)
                return results
        except Exception as exc:
            logger.warning("트레이스 조회 실패: %s", exc)
            return []

    def get_metrics_summary(self) -> dict[str, Any]:
        """LMOps 관측성 종합 요약 통계를 반환함"""
        from .db import _connect

        try:
            with _connect(self.settings) as conn:
                self.init_schema(conn)
                cur = conn.execute(
                    """
                    SELECT
                        count(*) AS total_queries,
                        coalesce(avg(total_latency_ms), 0) AS avg_latency_ms,
                        coalesce(sum(total_tokens), 0) AS total_tokens,
                        count(*) FILTER (WHERE guardrail_action = 'block') AS blocked_queries,
                        count(*) FILTER (WHERE hallucination_risk = 'high') AS high_hallucination_queries
                    FROM rag_lmops_traces
                    """
                )
                trace_stats = cur.fetchone() or {}

                cur_fb = conn.execute(
                    """
                    SELECT
                        count(*) AS feedback_count,
                        coalesce(avg(rating), 0) AS avg_rating,
                        count(*) FILTER (WHERE thumbs > 0) AS thumbs_up,
                        count(*) FILTER (WHERE thumbs < 0) AS thumbs_down
                    FROM rag_lmops_feedback
                    """
                )
                fb_stats = cur_fb.fetchone() or {}

                total_q = int(trace_stats.get("total_queries", 0))
                blocked = int(trace_stats.get("blocked_queries", 0))

                return {
                    "total_queries": total_q,
                    "avg_latency_ms": round(float(trace_stats.get("avg_latency_ms", 0.0)), 1),
                    "total_tokens": int(trace_stats.get("total_tokens", 0)),
                    "estimated_cost_usd": 0.0,
                    "guardrail_block_rate": round(blocked / max(1, total_q), 3),
                    "high_hallucination_count": int(trace_stats.get("high_hallucination_queries", 0)),
                    "feedback_count": int(fb_stats.get("feedback_count", 0)),
                    "avg_rating": round(float(fb_stats.get("avg_rating", 0.0)), 2),
                    "thumbs_up": int(fb_stats.get("thumbs_up", 0)),
                    "thumbs_down": int(fb_stats.get("thumbs_down", 0)),
                }
        except Exception as exc:
            logger.warning("LMOps 요약 메트릭 계산 실패: %s", exc)
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "guardrail_block_rate": 0.0,
                "high_hallucination_count": 0,
                "feedback_count": 0,
                "avg_rating": 0.0,
                "thumbs_up": 0,
                "thumbs_down": 0,
            }

    def export_feedback_to_sft(
        self,
        output_path: str = "data/sft/feedback_curated_sft.json",
        min_rating: int = 4,
    ) -> dict[str, Any]:
        """긍정 피드백 또는 사용자가 수정한 고품질 QA를 SFT 학습용 데이터셋으로 내보냄"""
        from .db import _connect

        with _connect(self.settings) as conn:
            self.init_schema(conn)
            cur = conn.execute(
                """
                SELECT
                    t.query_text,
                    coalesce(f.corrected_answer, t.answer_text) AS final_answer
                FROM rag_lmops_traces t
                JOIN rag_lmops_feedback f ON t.id = f.trace_id
                WHERE (f.thumbs > 0 OR f.rating >= %s OR f.corrected_answer IS NOT NULL)
                  AND t.answer_text IS NOT NULL
                """,
                (min_rating,),
            )
            rows = cur.fetchall()

            samples = [
                {
                    "instruction": "다음 질의에 대해 신뢰할 수 있는 정확한 답변을 작성하십시오.",
                    "input": r["query_text"],
                    "output": r["final_answer"],
                }
                for r in rows
                if r["query_text"] and r["final_answer"]
            ]

            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            return {
                "exported_count": len(samples),
                "output_path": str(out_file),
            }


_GLOBAL_LMOPS: LmopsManager | None = None
_LMOPS_LOCK = threading.Lock()


def get_lmops(settings: Settings) -> LmopsManager:
    """싱글톤 LmopsManager 인스턴스를 반환함"""
    global _GLOBAL_LMOPS
    if _GLOBAL_LMOPS is None:
        with _LMOPS_LOCK:
            if _GLOBAL_LMOPS is None:
                _GLOBAL_LMOPS = LmopsManager(settings)
    return _GLOBAL_LMOPS
