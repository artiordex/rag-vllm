#!/usr/bin/env python3
# =============================================================================
# 파일명: watch_inbox.py
# 경로: scripts/watch_inbox.py
# 목적: 인박스 문서를 감지해 RAG API로 전송하고 처리 폴더로 이동함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""인박스 폴더(data/inbox)를 감시해 RAG에 적재하는 스크립트임

HWPX·PDF·DOCX·JSON 파일을 rag-vllm API로 전송하고 성공한 파일을
data/processed 폴더로 이동함
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import httpx

INBOX_DIR = Path(__file__).resolve().parent.parent / "data" / "inbox"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:11020/documents/file")

SUPPORTED_EXTS = {".pdf", ".hwpx", ".docx", ".json", ".html", ".txt"}


def process_file(file_path: Path) -> bool:
    """지원 파일을 RAG API에 업로드하고 성공한 파일을 처리 폴더로 이동함

    Args:
        file_path: 인박스에서 처리할 파일 경로임

    Returns:
        bool: 업로드와 이동을 모두 완료했으면 `True`임

    Caveats:
        API 호출 실패 파일은 인박스에 남겨 다음 감시 주기에 재시도함
    """
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return False

    print(f"[*] 새 문서 감지: {file_path.name}")
    try:
        with open(file_path, "rb") as f:
            res = httpx.post(
                API_URL,
                files={"file": (file_path.name, f.read())},
                timeout=60.0,
            )
        if res.status_code in {200, 201}:
            data = res.json()
            print(f"[+] 적재 완료: {file_path.name} (문서 ID: {data.get('document_id')}, 청크: {data.get('chunk_count')})")
            target = PROCESSED_DIR / file_path.name
            shutil.move(str(file_path), str(target))
            return True
        else:
            print(f"[!] 업로드 실패 ({res.status_code}): {res.text}")
            return False
    except Exception as exc:
        print(f"[!] 오류 발생 ({file_path.name}): {exc}")
        return False


def main() -> None:
    """인박스 폴더를 주기적으로 확인하는 워커를 실행함"""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(" [문서 AI 자동화] 인박스 감시 워커 가동 중...")
    print(f" - 감시 폴더:  {INBOX_DIR}")
    print(f" - 완료 폴더:  {PROCESSED_DIR}")
    print(f" - 대상 API:   {API_URL}")
    print("=" * 60)

    while True:
        try:
            for file_path in INBOX_DIR.iterdir():
                if file_path.is_file() and not file_path.name.startswith("."):
                    process_file(file_path)
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n감시 워커를 종료합니다.")
            break


if __name__ == "__main__":
    main()
