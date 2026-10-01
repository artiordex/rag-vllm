# =============================================================================
# 파일명: dashboard.py
# 경로: src/rag_vllm/dashboard.py
# 목적: 외부 정적 파일 없이 제공하는 내부 RAG POC 대시보드 생성함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""외부 정적 파일 없이 제공하는 내부 RAG POC 대시보드 생성함"""

from __future__ import annotations


def get_dashboard_html() -> str:
    """rag-vllm 서비스 상태와 문서·검색 기능을 제공하는 단일 페이지 HTML을 반환함"""
    return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG-vLLM 문서 AI & 지능형 검색 포털 (POC)</title>
  <!-- Tailwind CSS CDN 로드 영역 -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- 마크다운 렌더링용 Marked.js 로드 영역 -->
  <script src="https://cdn.jsdelivr.net/npm/marked@18.0.7/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.4.16/dist/purify.min.js"></script>
  <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    body { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif; }
    .tab-active { border-bottom: 2px solid #2563eb; color: #1d4ed8; font-weight: 600; }
    .prose-korean h1, .prose-korean h2, .prose-korean h3 { font-weight: 700; margin-top: 1rem; margin-bottom: 0.5rem; }
    .prose-korean p { margin-bottom: 0.5rem; line-height: 1.7; }
    .prose-korean ul, .prose-korean ol { margin-left: 1.25rem; margin-bottom: 0.5rem; }
    .prose-korean table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    .prose-korean th, .prose-korean td { border: 1px solid #e2e8f0; padding: 0.5rem 0.75rem; text-align: left; }
    .prose-korean th { background-color: #f8fafc; font-weight: 600; }
  </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col">

  <!-- 헤더 영역 -->
  <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center space-x-3">
          <div class="w-10 h-10 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold text-xl shadow-md">
            AI
          </div>
          <div>
            <div class="flex items-center space-x-2">
              <span class="text-lg font-bold tracking-tight text-slate-900">RAG-vLLM 문서 AI & 지능형 검색</span>
              <span class="px-2 py-0.5 text-xs font-semibold bg-blue-100 text-blue-800 rounded-full">사내 POC</span>
            </div>
            <p class="text-xs text-slate-500">공문서·보고서 자동 기안, HWPX/PDF 파싱 & 하이브리드 RRF 검색</p>
          </div>
        </div>

        <!-- 실시간 상태 표시 영역 -->
        <div class="hidden md:flex items-center space-x-4 text-xs">
          <div class="flex items-center space-x-1.5 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200">
            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" id="status-dot"></span>
            <span class="text-slate-600 font-medium" id="vllm-badge">vLLM 11435 : 연결 확인 중</span>
          </div>
          <div class="flex items-center space-x-1.5 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200">
            <span class="text-slate-500">문서</span>
            <span class="font-bold text-slate-800" id="stat-docs">-</span>
            <span class="text-slate-300">|</span>
            <span class="text-slate-500">청크</span>
            <span class="font-bold text-slate-800" id="stat-chunks">-</span>
          </div>
          <span class="text-slate-500 font-medium">API 문서 비활성화</span>
        </div>
      </div>

      <!-- 탐색 탭 영역 -->
      <nav class="flex space-x-8 -mb-px">
        <button onclick="switchTab('search')" id="tab-search" class="tab-active py-3 px-1 text-sm font-medium border-b-2 flex items-center space-x-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
          <span>스마트 검색 & RAG 질의</span>
        </button>
        <button onclick="switchTab('draft')" id="tab-draft" class="text-slate-500 hover:text-slate-700 py-3 px-1 text-sm font-medium border-b-2 border-transparent flex items-center space-x-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
          <span>공문서·보고서 자동 기안</span>
        </button>
        <button onclick="switchTab('extract')" id="tab-extract" class="text-slate-500 hover:text-slate-700 py-3 px-1 text-sm font-medium border-b-2 border-transparent flex items-center space-x-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
          <span>정형 데이터 추출</span>
        </button>
        <button onclick="switchTab('vault')" id="tab-vault" class="text-slate-500 hover:text-slate-700 py-3 px-1 text-sm font-medium border-b-2 border-transparent flex items-center space-x-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
          <span>문서 보관소 & 업로드</span>
        </button>
        <button onclick="switchTab('system')" id="tab-system" class="text-slate-500 hover:text-slate-700 py-3 px-1 text-sm font-medium border-b-2 border-transparent flex items-center space-x-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
          <span>시스템 & n8n 연동</span>
        </button>
        <button onclick="switchTab('lmops')" id="tab-lmops" class="text-slate-500 hover:text-slate-700 py-3 px-1 text-sm font-medium border-b-2 border-transparent flex items-center space-x-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
          <span>LMOps & 가드레일</span>
        </button>
      </nav>
      <div class="flex flex-col sm:flex-row sm:items-center gap-2 border-t border-slate-100 py-2 text-xs">
        <label for="api-key-input" class="font-medium text-slate-600">RAG API 키</label>
        <input id="api-key-input" type="password" autocomplete="new-password" placeholder="키 설정 시 입력"
               class="w-full sm:w-72 px-2.5 py-1.5 rounded border border-slate-300 text-xs">
        <button onclick="saveApiKey()" class="px-3 py-1.5 rounded bg-slate-800 text-white hover:bg-slate-700">현재 탭에 저장</button>
        <button onclick="clearApiKey()" class="px-3 py-1.5 rounded bg-slate-100 text-slate-700 hover:bg-slate-200">지우기</button>
        <span id="api-key-status" class="text-slate-500">키는 이 페이지의 메모리에만 유지됩니다.</span>
      </div>
    </div>
  </header>

  <!-- 본문 컨테이너 -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">

    <!-- ========================================== -->
    <!-- 탭 1: 지능형 검색과 RAG 질의응답 -->
    <!-- ========================================== -->
    <section id="panel-search" class="space-y-6">
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 class="text-lg font-bold text-slate-900 mb-2">지능형 문서 검색 & RAG 질의응답</h2>
        <p class="text-sm text-slate-500 mb-4">공공 및 사내 규정 문서를 바탕으로 근거를 인용하여 정밀한 행정 답변을 생성합니다.</p>

        <!-- 검색 제어 영역 -->
        <div class="space-y-4">
          <div class="flex flex-col sm:flex-row gap-3">
            <div class="flex-1 relative">
              <input type="text" id="search-input" placeholder="사내 문서 및 규정에 대해 질문하세요... (예: 식품위생법 위반 시 영업정지 기준은?)"
                     class="w-full px-4 py-3 pl-11 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-slate-900 placeholder-slate-400 text-sm shadow-inner"
                     onkeydown="if(event.key === 'Enter') executeQuery();">
              <svg class="w-5 h-5 text-slate-400 absolute left-3.5 top-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </div>
            <button onclick="executeQuery()" id="btn-search" class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow transition flex items-center justify-center space-x-2 text-sm">
              <span>질문하기</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
            </button>
          </div>

          <!-- 필터·검색 모드 배지 영역 -->
          <div class="flex flex-wrap items-center gap-4 text-xs text-slate-600 pt-1">
            <div class="flex items-center space-x-2">
              <span class="font-medium text-slate-700">검색 방식:</span>
              <label class="inline-flex items-center space-x-1 cursor-pointer">
                <input type="radio" name="search-mode" value="hybrid" checked class="text-blue-600 focus:ring-blue-500">
                <span class="font-semibold text-blue-700">하이브리드 RRF (추천)</span>
              </label>
              <label class="inline-flex items-center space-x-1 cursor-pointer">
                <input type="radio" name="search-mode" value="dense" class="text-blue-600 focus:ring-blue-500">
                <span>벡터만 (Dense)</span>
              </label>
            </div>

            <div class="flex items-center space-x-2">
              <span class="font-medium text-slate-700">참고 문서 수 (Top-K):</span>
              <select id="search-top-k" class="bg-slate-50 border border-slate-300 rounded px-2 py-1 text-xs">
                <option value="3">3개</option>
                <option value="5" selected>5개 (기본)</option>
                <option value="10">10개 (정밀)</option>
              </select>
            </div>

            <div class="flex items-center space-x-2">
              <span class="font-medium text-slate-700">문서 제한:</span>
              <select id="search-doc-filter" class="bg-slate-50 border border-slate-300 rounded px-2 py-1 text-xs max-w-xs truncate">
                <option value="">전체 문서 대상</option>
              </select>
            </div>
          </div>

          <!-- 추천 프롬프트 영역 -->
          <div class="flex flex-wrap items-center gap-2 pt-2">
            <span class="text-xs text-slate-400">추천 질문:</span>
            <button onclick="setQuery('식품위생법 위반 시 영업정지 및 과태료 행정처분 기준은 무엇인가요?')" class="text-xs bg-slate-100 hover:bg-blue-50 hover:text-blue-700 text-slate-600 px-2.5 py-1 rounded-full border border-slate-200 transition">
              식품위생 행정처분 기준
            </button>
            <button onclick="setQuery('2026년 공공데이터 품질관리 실태평가 대상과 중점 추진 지표는?')" class="text-xs bg-slate-100 hover:bg-blue-50 hover:text-blue-700 text-slate-600 px-2.5 py-1 rounded-full border border-slate-200 transition">
              공공데이터 품질관리 지침
            </button>
            <button onclick="setQuery('생성형 AI 공공업무 도입 시 개인정보 및 보안 준수 수칙을 알려줘')" class="text-xs bg-slate-100 hover:bg-blue-50 hover:text-blue-700 text-slate-600 px-2.5 py-1 rounded-full border border-slate-200 transition">
              생성형 AI 보안 가이드라인
            </button>
          </div>
        </div>
      </div>

      <!-- 검색 결과 영역 -->
      <div id="search-results-box" class="hidden space-y-6">
        <!-- AI 답변 카드 -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div class="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
            <div class="flex items-center space-x-2">
              <span class="w-3 h-3 rounded-full bg-blue-600"></span>
              <h3 class="font-bold text-slate-900">vLLM AI 종합 답변</h3>
            </div>
            <span class="text-xs text-slate-400" id="search-elapsed"></span>
          </div>
          <div id="search-answer-content" class="prose-korean text-slate-800 text-sm leading-relaxed">
            <!-- 답변을 렌더링하는 영역 -->
          </div>
        </div>

        <!-- 검색 청크 카드 영역 -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <h3 class="font-bold text-slate-900 mb-4 flex items-center space-x-2">
            <span>참조된 원문 근거 (Citations)</span>
            <span class="text-xs font-normal text-slate-400" id="search-citation-count"></span>
          </h3>
          <div id="search-sources-list" class="space-y-3">
            <!-- 출처 카드를 삽입하는 영역 -->
          </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- 탭 2: 공식 문서 초안 스튜디오 -->
    <!-- ========================================== -->
    <section id="panel-draft" class="hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- 입력 양식 -->
        <div class="lg:col-span-5 bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-4">
          <h2 class="text-lg font-bold text-slate-900">공문서·보고서 기안 양식 설정</h2>
          <p class="text-sm text-slate-500">사내 RAG 근거를 바탕으로 공문서 초안을 작성합니다. 생성 결과의 근거와 서식 적합성은 담당자가 검토해야 합니다.</p>

          <div>
            <label class="block text-xs font-bold text-slate-700 mb-1">문체 및 서식 스타일</label>
            <select id="draft-style" class="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500">
              <option value="공문서_개조식" selected>공문서 개조식 초안 (~추진함, 1. 가. (1) 번호체계)</option>
              <option value="보고서_서술형">전문 정책 분석 보고서 (서술형, 현황-문제점-대응방안)</option>
              <option value="요약표">경영진 보고용 마크다운 요약표</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-bold text-slate-700 mb-1">기안 문서 제목 / 주제</label>
            <input type="text" id="draft-title" placeholder="예: 2026년 공공데이터 품질관리 실태평가 세부 시행계획"
                   class="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500">
          </div>

          <div>
            <label class="block text-xs font-bold text-slate-700 mb-1">추가 작성 지침 / 중점 사항 (선택)</label>
            <textarea id="draft-instructions" rows="4" placeholder="예: 1. 점검 대상 기관과 중점 지표를 명시할 것.&#10;2. 위반 시 조치계획과 향후 일정표를 포함할 것."
                      class="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500"></textarea>
          </div>

          <!-- 빠른 템플릿 -->
          <div>
            <span class="text-xs font-medium text-slate-400">빠른 예시 템플릿:</span>
            <div class="flex flex-wrap gap-1.5 mt-1.5">
              <button onclick="fillDraftTemplate('공문서_개조식', '식품위생 위반업소 행정처분 및 시정명령 통보 건', '1. 위반 사실과 관련 법령을 명시할 것.\n2. 영업정지 처분 기간 및 의견제출 기한을 포함할 것.')"
                      class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2 py-1 rounded">식품위생 처분 통보</button>
              <button onclick="fillDraftTemplate('공문서_개조식', '2026년도 데이터 품질진단 실태점검 추진 계획', '추진배경, 중점 진단항목, 단계별 추진일정을 개조식으로 명시할 것.')"
                      class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2 py-1 rounded">품질점검 추진계획</button>
              <button onclick="fillDraftTemplate('요약표', '행정처분 기준 및 과태료 부과 종합 비교표', '위반행위, 관련법조항, 1차/2차/3차 처분내용을 표로 작성할 것.')"
                      class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2 py-1 rounded">처분기준 비교표</button>
            </div>
          </div>

          <button onclick="executeDraft()" id="btn-draft" class="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg shadow transition flex items-center justify-center space-x-2 text-sm mt-4">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            <span>공문서 초안 자동 작성</span>
          </button>
        </div>

        <!-- 출력 미리보기 -->
        <div class="lg:col-span-7 bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col">
          <div class="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
            <h3 class="font-bold text-slate-900 flex items-center space-x-2">
              <span class="w-3 h-3 rounded-full bg-indigo-600"></span>
              <span id="draft-result-heading">기안 문서 미리보기</span>
            </h3>
            <div class="flex items-center space-x-2">
              <button onclick="copyDraftContent()" id="btn-copy-draft" class="px-2.5 py-1 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded flex items-center space-x-1">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                <span>복사</span>
              </button>
              <button onclick="downloadDraftMd()" class="px-2.5 py-1 text-xs font-medium text-indigo-600 bg-indigo-50 hover:bg-indigo-100 rounded flex items-center space-x-1">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
                <span>.md 저장</span>
              </button>
            </div>
          </div>

          <div id="draft-output" class="flex-1 bg-slate-50 border border-slate-200 rounded-lg p-5 overflow-auto prose-korean text-sm leading-relaxed text-slate-800 min-h-[400px]">
            <p class="text-slate-400 italic text-center my-32">좌측에서 기안 제목과 양식을 설정한 후 '공문서 초안 자동 작성'을 누르세요.</p>
          </div>

          <!-- 초안에서 참조한 근거 -->
          <div id="draft-sources-container" class="mt-4 pt-3 border-t border-slate-100 hidden">
            <h4 class="text-xs font-bold text-slate-600 mb-2">기안 작성 시 인용된 사내 문서 근거:</h4>
            <div id="draft-sources-list" class="space-y-1.5 text-xs text-slate-600"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- 탭 3: 구조화 항목 추출 -->
    <!-- ========================================== -->
    <section id="panel-extract" class="hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- 설정 영역 -->
        <div class="lg:col-span-5 bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-4">
          <h2 class="text-lg font-bold text-slate-900">정형 데이터 자동 추출</h2>
          <p class="text-sm text-slate-500">비정형 문서에서 행정 처분 내역, 사업 계획, 공문서 메타데이터 등을 정형 JSON으로 구조화합니다.</p>

          <div>
            <label class="block text-xs font-bold text-slate-700 mb-1">추출 스키마</label>
            <select id="extract-schema" class="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500">
              <option value="공문서_메타데이터">공문서 메타데이터 (문서번호, 시행일자, 발신/수신, 담당자, 요약)</option>
              <option value="행정처분_요약">행정처분 요약 (처분대상, 위반법령, 처분종류, 위반내용, 기간)</option>
              <option value="사업계획_요약">사업계획 요약 (사업명, 예산, 기간, 주관부서, 목표, 과업목록)</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-bold text-slate-700 mb-1">보관소 문서 선택 (우선)</label>
            <select id="extract-doc-select" class="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500">
              <option value="">-- 직접 텍스트 입력 사용 --</option>
            </select>
          </div>

          <div>
            <label class="block text-xs font-bold text-slate-700 mb-1">또는 텍스트 직접 입력</label>
            <textarea id="extract-text" rows="7" placeholder="문서 원문 텍스트를 붙여넣으세요..."
                      class="w-full px-3 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500"></textarea>
          </div>

          <button onclick="executeExtract()" id="btn-extract" class="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-medium rounded-lg shadow transition flex items-center justify-center space-x-2 text-sm">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
            <span>정형 데이터 추출 실행</span>
          </button>
        </div>

        <!-- 추출 결과 영역 -->
        <div class="lg:col-span-7 bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col">
          <div class="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
            <h3 class="font-bold text-slate-900 flex items-center space-x-2">
              <span class="w-3 h-3 rounded-full bg-emerald-600"></span>
              <span>구조화 데이터 추출 결과</span>
            </h3>
            <button onclick="copyExtractJson()" class="px-2.5 py-1 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded flex items-center space-x-1">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
              <span>JSON 복사</span>
            </button>
          </div>

          <div id="extract-output" class="flex-1 bg-slate-50 border border-slate-200 rounded-lg p-5 overflow-auto text-sm min-h-[400px]">
            <p class="text-slate-400 italic text-center my-32">좌측에서 문서 선택 또는 텍스트 입력 후 추출을 실행하세요.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- 탭 4: 문서 보관함과 업로드 -->
    <!-- ========================================== -->
    <section id="panel-vault" class="hidden space-y-6">
      <!-- 업로드 영역 -->
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 class="text-lg font-bold text-slate-900 mb-2">신규 사내 문서 인제스트 (Ingestion)</h2>
        <p class="text-sm text-slate-500 mb-4">HWPX, PDF, TXT, DOCX, CSV 파일을 업로드하면 추출 텍스트를 휴리스틱 진단·임베딩하고 pgvector에 보관합니다. 진단 점수는 공식 적합 판정이 아닙니다.</p>

        <div id="dropzone" class="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-8 text-center bg-slate-50 hover:bg-blue-50/40 transition cursor-pointer"
             onclick="document.getElementById('file-input').click();">
          <input type="file" id="file-input" class="hidden" onchange="handleFileUpload(event)">
          <div class="w-12 h-12 mx-auto mb-3 text-blue-600 bg-blue-100 rounded-full flex items-center justify-center">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
          </div>
          <p class="text-sm font-semibold text-slate-700">클릭하거나 파일을 여기로 끌어다 놓으세요</p>
          <p class="text-xs text-slate-400 mt-1">지원 포맷: .hwpx, .pdf, .txt, .md, .docx, .csv (최대 20MB)</p>
          <div id="upload-status" class="mt-3 text-xs font-semibold hidden"></div>
        </div>
      </div>

      <!-- 문서 목록 표 -->
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="font-bold text-slate-900">등록된 사내 문서 목록</h3>
            <p class="text-xs text-slate-500">현재 pgvector에 벡터화되어 검색 가능한 문서 목록입니다.</p>
          </div>
          <button onclick="loadDocuments()" class="px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg flex items-center space-x-1">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
            <span>새로고침</span>
          </button>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-sm text-slate-700">
            <thead class="bg-slate-50 text-xs text-slate-500 uppercase border-b border-slate-200">
              <tr>
                <th class="px-4 py-3">문서명</th>
                <th class="px-3 py-3">종류</th>
                <th class="px-3 py-3">청크 수</th>
                <th class="px-3 py-3">품질 점수</th>
                <th class="px-4 py-3">등록일시</th>
                <th class="px-4 py-3 text-right">관리</th>
              </tr>
            </thead>
            <tbody id="docs-table-body" class="divide-y divide-slate-100">
              <tr>
                <td colspan="6" class="px-4 py-8 text-center text-slate-400">문서 목록을 불러오는 중...</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- 탭 5: 시스템과 n8n 연동 -->
    <!-- ========================================== -->
    <section id="panel-system" class="hidden space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- 시스템 상태 카드 -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-4">
          <h2 class="text-lg font-bold text-slate-900">엔진 상태 및 하드웨어 사양</h2>
          
          <div class="space-y-3 text-sm">
            <div class="flex justify-between items-center py-2 border-b border-slate-100">
              <span class="text-slate-500">FastAPI RAG 서비스 포트</span>
              <span class="font-mono font-semibold text-slate-800">11020 (AI 전용 포트)</span>
            </div>
            <div class="flex justify-between items-center py-2 border-b border-slate-100">
              <span class="text-slate-500">vLLM 호스트 서버 주소</span>
              <span class="font-mono font-semibold text-slate-800">http://127.0.0.1:11435/v1</span>
            </div>
            <div class="flex justify-between items-center py-2 border-b border-slate-100">
              <span class="text-slate-500">vLLM 서빙 모델</span>
              <span class="font-mono font-semibold text-blue-700" id="sys-vllm-model">rag-vllm-model</span>
            </div>
            <div class="flex justify-between items-center py-2 border-b border-slate-100">
              <span class="text-slate-500">임베딩 모델 및 차원</span>
              <span class="font-mono font-semibold text-slate-800" id="sys-embedding">BAAI/bge-m3 (1024차원)</span>
            </div>
            <div class="flex justify-between items-center py-2 border-b border-slate-100">
              <span class="text-slate-500">PostgreSQL (pgvector) 포트</span>
              <span class="font-mono font-semibold text-slate-800">55433 (ragvllm)</span>
            </div>
            <div class="flex justify-between items-center py-2">
              <span class="text-slate-500">GPU VRAM 설정 (RTX 5060 Ti)</span>
              <span class="font-mono font-semibold text-emerald-600">기본 utilization 0.75 (설정 상한 약 12 GiB)</span>
            </div>
          </div>
        </div>

        <!-- n8n 연동 안내 카드 -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-4">
          <h2 class="text-lg font-bold text-slate-900">n8n 자동화 파이프라인 연동 규격</h2>
          <p class="text-sm text-slate-500">사내 포털이나 n8n 워크플로우에서 HTTP Request 노드로 바로 호출할 수 있습니다.</p>

          <div class="space-y-3 text-xs">
            <div>
              <span class="font-bold text-slate-700 block mb-1">1. 하이브리드 RAG 질의 노드 (POST /query)</span>
              <pre class="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto">curl -X POST http://127.0.0.1:11020/query \
  -H "Content-Type: application/json" \
  -d '{"question": "영업정지 행정처분 기준은?", "search_mode": "hybrid", "top_k": 5}'</pre>
            </div>
            <div>
              <span class="font-bold text-slate-700 block mb-1">2. 공문서 자동 기안 노드 (POST /draft)</span>
              <pre class="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto">curl -X POST http://127.0.0.1:11020/draft \
  -H "Content-Type: application/json" \
  -d '{"title": "식품위생 위반 시정명령", "style": "공문서_개조식"}'</pre>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ========================================== -->
    <!-- 탭 6: LMOps, 가드레일 & 정량 평가 패널 -->
    <!-- ========================================== -->
    <section id="panel-lmops" class="hidden space-y-6">
      <!-- 핵심 KPI 지표 카드 그리드 -->
      <div class="grid grid-cols-2 md:grid-cols-6 gap-4">
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider">누적 질의</div>
          <div class="text-2xl font-bold text-slate-900 mt-1" id="lmops-total-queries">-</div>
          <div class="text-xs text-blue-600 mt-1 font-medium">100% 로컬 처리</div>
        </div>
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider">평균 지연시간</div>
          <div class="text-2xl font-bold text-slate-900 mt-1"><span id="lmops-avg-latency">-</span> <span class="text-xs font-normal text-slate-500">ms</span></div>
          <div class="text-xs text-slate-500 mt-1">E2E 파이프라인</div>
        </div>
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider">누적 소비 토큰</div>
          <div class="text-2xl font-bold text-slate-900 mt-1" id="lmops-total-tokens">-</div>
          <div class="text-xs text-emerald-600 mt-1 font-medium">추가 비용: $0.00</div>
        </div>
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider">가드레일 차단율</div>
          <div class="text-2xl font-bold text-slate-900 mt-1"><span id="lmops-block-rate">-</span>%</div>
          <div class="text-xs text-amber-600 mt-1 font-medium">인젝션/탈옥 차단</div>
        </div>
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider">고환각 위험 건</div>
          <div class="text-2xl font-bold text-slate-900 mt-1" id="lmops-high-risk">-</div>
          <div class="text-xs text-slate-500 mt-1">문맥 미지원 진술</div>
        </div>
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
          <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider">피드백 평점</div>
          <div class="text-2xl font-bold text-slate-900 mt-1"><span id="lmops-avg-rating">-</span> <span class="text-xs font-normal text-slate-500">/ 5.0</span></div>
          <div class="text-xs text-slate-500 mt-1" id="lmops-feedback-count">-건 수집됨</div>
        </div>
      </div>

      <!-- 상단 컨트롤 및 SFT 내보내기 배너 -->
      <div class="bg-gradient-to-r from-blue-900 to-indigo-900 rounded-xl p-5 text-white flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-sm">
        <div>
          <h3 class="text-base font-bold">지속적 모델 개선 (Continuous SFT & LoRA Data Curation)</h3>
          <p class="text-xs text-blue-200 mt-0.5">사용자 추천(Thumbs Up) 및 평점 4점 이상의 고품질 질의응답을 SFT 파인튜닝 데이터셋으로 즉시 변환합니다.</p>
        </div>
        <div class="flex items-center space-x-2">
          <button onclick="exportCuratedSft()" class="px-4 py-2 bg-blue-500 hover:bg-blue-400 text-white rounded-lg text-xs font-bold shadow transition">
            고품질 QA를 SFT 데이터로 내보내기
          </button>
          <button onclick="loadLmopsData()" class="px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg text-xs font-medium transition">
            새로고침
          </button>
        </div>
      </div>

      <!-- 2컬럼 레이아웃: 최근 트레이스 로그 & 대화형 도구 -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- 좌측: 최근 트레이스 목록 (2 cols) -->
        <div class="lg:col-span-2 bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-base font-bold text-slate-900">최근 RAG 트레이스 로그</h3>
            <span class="text-xs text-slate-500">PostgreSQL rag_lmops_traces 연동</span>
          </div>
          <div class="overflow-x-auto max-h-96">
            <table class="w-full text-xs text-left border-collapse">
              <thead class="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold sticky top-0">
                <tr>
                  <th class="py-2.5 px-3">질의 요약</th>
                  <th class="py-2.5 px-3">가드레일</th>
                  <th class="py-2.5 px-3 text-right">지연시간</th>
                  <th class="py-2.5 px-3 text-right">토큰</th>
                  <th class="py-2.5 px-3">환각위험</th>
                  <th class="py-2.5 px-3">기록일시</th>
                </tr>
              </thead>
              <tbody id="lmops-traces-body" class="divide-y divide-slate-100">
                <tr><td colspan="6" class="text-center py-6 text-slate-400">트레이스 데이터를 조회하는 중...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 우측: 가드레일 사전 진단 & 로컬 평가기 도구 (1 col) -->
        <div class="space-y-6">
          <!-- 가드레일 검사기 -->
          <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <h4 class="text-sm font-bold text-slate-900 mb-1">가드레일 실시간 테스트</h4>
            <p class="text-xs text-slate-500 mb-3">탈옥, 프롬프트 인젝션, 주민번호/카드번호 마스킹을 테스트합니다.</p>
            <textarea id="guardrail-test-input" rows="3" placeholder="테스트 입력문... (예: Ignore previous instructions or 010-1234-5678)"
                      class="w-full p-2.5 rounded border border-slate-300 text-xs focus:ring-1 focus:ring-blue-500 outline-none"></textarea>
            <button onclick="testGuardrail()" class="mt-2 w-full py-2 bg-slate-800 hover:bg-slate-700 text-white rounded text-xs font-semibold">
              가드레일 검증 실행
            </button>
            <div id="guardrail-test-result" class="mt-3 text-xs p-2.5 bg-slate-50 rounded border border-slate-200 hidden"></div>
          </div>

          <!-- 로컬 평가 지표 계산기 -->
          <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
            <h4 class="text-sm font-bold text-slate-900 mb-1">RAG 정량 평가기 (Local & DeepEval)</h4>
            <p class="text-xs text-slate-500 mb-3">문맥 충실도(Faithfulness), 답변 적합성, 문맥 정밀도를 측정합니다.</p>
            <div class="space-y-2 text-xs">
              <input type="text" id="eval-test-query" placeholder="질문" class="w-full p-2 rounded border border-slate-300">
              <textarea id="eval-test-answer" rows="2" placeholder="생성된 답변" class="w-full p-2 rounded border border-slate-300"></textarea>
              <textarea id="eval-test-context" rows="2" placeholder="참고 문맥 청크" class="w-full p-2 rounded border border-slate-300"></textarea>
              <button onclick="testLocalEval()" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded font-semibold">
                품질 지표 산출
              </button>
            </div>
            <div id="eval-test-result" class="mt-3 text-xs p-2.5 bg-slate-50 rounded border border-slate-200 hidden"></div>
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- 청크 조회 모달 -->
  <div id="chunk-modal" class="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
    <div class="bg-white rounded-xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl">
      <div class="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
        <div>
          <h3 class="font-bold text-slate-900" id="modal-doc-title">문서 청크 분할 열람</h3>
          <p class="text-xs text-slate-500" id="modal-doc-subtitle"></p>
        </div>
        <button onclick="closeChunkModal()" class="text-slate-400 hover:text-slate-600 p-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="p-6 overflow-y-auto space-y-4 flex-1" id="modal-chunks-body">
        <!-- 청크 목록 -->
      </div>
      <div class="px-6 py-3 border-t border-slate-200 bg-slate-50 text-right">
        <button onclick="closeChunkModal()" class="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-700 text-sm font-medium rounded-lg">닫기</button>
      </div>
    </div>
  </div>

  <footer class="bg-white border-t border-slate-200 py-4 mt-auto">
    <div class="max-w-7xl mx-auto px-4 text-center text-xs text-slate-400">
      RAG-vLLM Document AI Enterprise Architecture &bull; Dedicated Port 11020 &bull; vLLM 11435 &bull; pgvector 55433
    </div>
  </footer>

  <!-- 대시보드 JavaScript 애플리케이션 -->
  <script>
    let currentRawDraft = "";
    let currentExtractJson = "";
    let cachedDocuments = [];

    let ragApiKey = '';

    function saveApiKey() {
      const input = document.getElementById('api-key-input');
      const key = input.value.trim();
      if (key) {
        ragApiKey = key;
        input.value = '';
        document.getElementById('api-key-status').innerText = '키를 이 페이지에 적용했습니다.';
      } else {
        clearApiKey();
        return;
      }
      checkHealth();
      loadDocuments();
    }

    function clearApiKey() {
      ragApiKey = '';
      document.getElementById('api-key-input').value = '';
      document.getElementById('api-key-status').innerText = '페이지에서 키를 지웠습니다.';
    }

    async function apiFetch(input, options = {}) {
      const headers = new Headers(options.headers || {});
      if (ragApiKey) headers.set('X-API-Key', ragApiKey);
      const response = await fetch(input, { ...options, headers });
      if (response.status === 401) {
        document.getElementById('api-key-status').innerText = 'API 키를 확인하고 다시 저장해 주세요.';
      }
      return response;
    }

    function renderMarkdown(text) {
      const source = String(text || '');
      if (window.marked && window.DOMPurify) {
        return DOMPurify.sanitize(marked.parse(source), {
          ALLOWED_TAGS: ['a', 'b', 'blockquote', 'br', 'code', 'del', 'div', 'em', 'h1', 'h2', 'h3', 'h4', 'hr', 'i', 'li', 'ol', 'p', 'pre', 'span', 'strong', 'table', 'tbody', 'td', 'th', 'thead', 'tr', 'ul'],
          ALLOWED_ATTR: ['href', 'title'],
        });
      }
      return `<pre class="whitespace-pre-wrap">${escapeHtml(source)}</pre>`;
    }

    // 탭 전환 처리함
    function switchTab(tabName) {
      const tabs = ['search', 'draft', 'extract', 'vault', 'system', 'lmops'];
      tabs.forEach(t => {
        const btn = document.getElementById(`tab-${t}`);
        const panel = document.getElementById(`panel-${t}`);
        if (t === tabName) {
          btn.className = "tab-active py-3 px-1 text-sm font-medium border-b-2 flex items-center space-x-2";
          panel.classList.remove('hidden');
        } else {
          btn.className = "text-slate-500 hover:text-slate-700 py-3 px-1 text-sm font-medium border-b-2 border-transparent flex items-center space-x-2";
          panel.classList.add('hidden');
        }
      });
      if (tabName === 'vault' || tabName === 'search' || tabName === 'draft' || tabName === 'extract') {
        loadDocuments();
      }
      if (tabName === 'system') {
        loadStats();
      }
      if (tabName === 'lmops') {
        loadLmopsData();
      }
    }

    // 추천 질의 입력을 보조함
    function setQuery(text) {
      document.getElementById('search-input').value = text;
      executeQuery();
    }

    // RAG 질의를 실행함
    async function executeQuery() {
      const question = document.getElementById('search-input').value.trim();
      if (!question) return;

      const btn = document.getElementById('btn-search');
      btn.disabled = true;
      btn.innerHTML = `<span>검색 중...</span>`;

      const mode = document.querySelector('input[name="search-mode"]:checked').value;
      const topK = parseInt(document.getElementById('search-top-k').value, 10);
      const docId = document.getElementById('search-doc-filter').value || null;

      const startTime = performance.now();
      try {
        const res = await apiFetch('/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: question,
            search_mode: mode,
            top_k: topK,
            document_id: docId,
            use_llm: true
          })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || '검색 요청 실패');
        }

        const data = await res.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

        document.getElementById('search-results-box').classList.remove('hidden');
        document.getElementById('search-elapsed').innerText = `소요 시간: ${elapsed}초 &bull; 검색 모드: ${mode === 'hybrid' ? '하이브리드 RRF' : 'Dense 벡터'}`;
        document.getElementById('search-answer-content').innerHTML = renderMarkdown(data.answer || '답변을 생성하지 못했습니다.');

        // 인용 근거를 렌더링함
        const sourcesList = document.getElementById('search-sources-list');
        sourcesList.innerHTML = '';
        document.getElementById('search-citation-count').innerText = `총 ${data.sources.length}개 청크 인용`;

        data.sources.forEach(src => {
          const scoreDisplay = src.score ? (src.score * 100).toFixed(1) + '%' : '-';
          const card = document.createElement('div');
          card.className = "p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-1.5";
          card.innerHTML = `
            <div class="flex items-center justify-between">
              <span class="font-bold text-slate-800">[${src.rank}] ${escapeHtml(src.source_name)} (청크 #${src.chunk_index})</span>
              <div class="flex items-center space-x-2">
                <span class="px-2 py-0.5 bg-blue-100 text-blue-700 font-semibold rounded">점수 ${scoreDisplay}</span>
                <span class="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded">품질 ${src.quality_score || 100}점</span>
              </div>
            </div>
            <p class="text-slate-600 line-clamp-3 hover:line-clamp-none transition-all cursor-pointer bg-white p-2 rounded border border-slate-100">${escapeHtml(src.text)}</p>
          `;
          sourcesList.appendChild(card);
        });

      } catch (err) {
        alert('검색 중 오류 발생: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<span>질문하기</span><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>`;
      }
    }

    // 초안 입력 양식을 채움
    function fillDraftTemplate(style, title, instructions) {
      document.getElementById('draft-style').value = style;
      document.getElementById('draft-title').value = title;
      document.getElementById('draft-instructions').value = instructions;
    }

    // 공식 문서 초안 생성을 실행함
    async function executeDraft() {
      const title = document.getElementById('draft-title').value.trim();
      if (!title) {
        alert('기안 문서 제목을 입력하세요.');
        return;
      }
      const style = document.getElementById('draft-style').value;
      const instructions = document.getElementById('draft-instructions').value.trim();

      const btn = document.getElementById('btn-draft');
      btn.disabled = true;
      btn.innerHTML = `<span>공문서 작성 중 (vLLM 추론)...</span>`;

      const outputBox = document.getElementById('draft-output');
      outputBox.innerHTML = `<div class="text-center my-32"><div class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-indigo-500 border-t-transparent"></div><p class="mt-3 text-slate-500 text-xs">관련 사내 문서를 하이브리드 검색하고 행정 양식에 맞춰 작성 중입니다...</p></div>`;

      try {
        const res = await apiFetch('/draft', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: title,
            style: style,
            instructions: instructions,
            top_k: 5
          })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || '기안 작성 요청 실패');
        }

        const data = await res.json();
        currentRawDraft = data.content;
        outputBox.innerHTML = renderMarkdown(data.content);
        document.getElementById('draft-result-heading').innerText = `기안 문서: ${data.title} (${data.style})`;

        // 참조 근거를 렌더링함
        const sourcesContainer = document.getElementById('draft-sources-container');
        const sourcesList = document.getElementById('draft-sources-list');
        sourcesList.innerHTML = '';
        if (data.sources && data.sources.length > 0) {
          sourcesContainer.classList.remove('hidden');
          data.sources.forEach(s => {
            const item = document.createElement('div');
            item.className = "flex items-center space-x-2";
            item.innerHTML = `
              <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full"></span>
              <span class="font-medium text-slate-700">${escapeHtml(s.source_name)} (청크 #${s.chunk_index})</span>
            `;
            sourcesList.appendChild(item);
          });
        } else {
          sourcesContainer.classList.add('hidden');
        }

      } catch (err) {
        outputBox.innerHTML = `<p class="text-red-500">오류 발생: ${escapeHtml(err.message)}</p>`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg><span>공문서 초안 자동 작성</span>`;
      }
    }

    function copyDraftContent() {
      if (!currentRawDraft) return;
      navigator.clipboard.writeText(currentRawDraft).then(() => {
        const btn = document.getElementById('btn-copy-draft');
        const originalText = btn.innerHTML;
        btn.innerHTML = `<span>복사됨!</span>`;
        setTimeout(() => btn.innerHTML = originalText, 1500);
      });
    }

    function downloadDraftMd() {
      if (!currentRawDraft) return;
      const title = document.getElementById('draft-title').value.trim() || '기안문서';
      const blob = new Blob([currentRawDraft], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title.replace(/\\s+/g, '_')}.md`;
      a.click();
      URL.revokeObjectURL(url);
    }

    // 구조화 항목 추출을 실행함
    async function executeExtract() {
      const schema = document.getElementById('extract-schema').value;
      const docId = document.getElementById('extract-doc-select').value || null;
      const text = document.getElementById('extract-text').value.trim() || null;

      if (!docId && !text) {
        alert('보관소 문서를 선택하거나 텍스트를 입력하세요.');
        return;
      }

      const btn = document.getElementById('btn-extract');
      btn.disabled = true;
      btn.innerHTML = `<span>구조화 추출 중...</span>`;

      const outputBox = document.getElementById('extract-output');
      outputBox.innerHTML = `<div class="text-center my-32"><div class="inline-block animate-spin rounded-full h-8 w-8 border-4 border-emerald-500 border-t-transparent"></div><p class="mt-3 text-slate-500 text-xs">vLLM을 통해 비정형 텍스트에서 정형 스키마를 추출하고 있습니다...</p></div>`;

      try {
        const res = await apiFetch('/documents/extract', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            schema_type: schema,
            document_id: docId,
            text: text
          })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || '추출 실패');
        }

        const data = await res.json();
        currentExtractJson = JSON.stringify(data.extracted_data, null, 2);

        // 결과 카드와 JSON을 렌더링함
        let cardHtml = `<div class="space-y-4">`;
        cardHtml += `<div class="flex items-center justify-between"><span class="font-bold text-slate-800">문서: ${escapeHtml(data.source_name)}</span><span class="text-xs bg-emerald-100 text-emerald-800 px-2.5 py-0.5 rounded-full font-semibold">${escapeHtml(data.schema_type)}</span></div>`;
        cardHtml += `<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">`;

        for (const [key, value] of Object.entries(data.extracted_data)) {
          let valDisplay = typeof value === 'object' ? JSON.stringify(value, null, 2) : value;
          cardHtml += `
            <div class="p-3 bg-white border border-slate-200 rounded-lg">
              <span class="text-xs font-semibold text-slate-500 block mb-1">${escapeHtml(key)}</span>
              <span class="text-sm font-medium text-slate-800 break-words">${escapeHtml(String(valDisplay))}</span>
            </div>
          `;
        }
        cardHtml += `</div>`;
        cardHtml += `<div class="mt-4"><span class="text-xs font-bold text-slate-600 mb-1 block">원문 JSON 구조:</span><pre class="bg-slate-900 text-emerald-400 p-3 rounded-lg text-xs font-mono overflow-x-auto">${escapeHtml(currentExtractJson)}</pre></div>`;
        cardHtml += `</div>`;

        outputBox.innerHTML = cardHtml;

      } catch (err) {
        outputBox.innerHTML = `<p class="text-red-500">오류 발생: ${escapeHtml(err.message)}</p>`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg><span>정형 데이터 추출 실행</span>`;
      }
    }

    function copyExtractJson() {
      if (!currentExtractJson) return;
      navigator.clipboard.writeText(currentExtractJson).then(() => {
        alert('추출된 JSON이 클립보드에 복사되었습니다.');
      });
    }

    // API에서 문서 목록을 불러옴
    async function loadDocuments() {
      try {
        const res = await apiFetch('/documents?limit=100');
        if (!res.ok) return;
        const data = await res.json();
        cachedDocuments = data.items || [];

        // 헤더 통계를 갱신함
        document.getElementById('stat-docs').innerText = data.total;
        const totalChunks = cachedDocuments.reduce((acc, cur) => acc + (cur.chunk_count || 0), 0);
        document.getElementById('stat-chunks').innerText = totalChunks;

        // 검색·추출 탭의 문서 필터를 갱신함
        const searchDocFilter = document.getElementById('search-doc-filter');
        const extractDocSelect = document.getElementById('extract-doc-select');
        const curSearchVal = searchDocFilter.value;
        const curExtractVal = extractDocSelect.value;

        searchDocFilter.innerHTML = '<option value="">전체 문서 대상</option>';
        extractDocSelect.innerHTML = '<option value="">-- 직접 텍스트 입력 사용 --</option>';

        cachedDocuments.forEach(doc => {
          searchDocFilter.innerHTML += `<option value="${doc.id}">${escapeHtml(doc.source_name)}</option>`;
          extractDocSelect.innerHTML += `<option value="${doc.id}">${escapeHtml(doc.source_name)}</option>`;
        });

        searchDocFilter.value = curSearchVal;
        extractDocSelect.value = curExtractVal;

        // 문서 표 본문을 렌더링함
        const tbody = document.getElementById('docs-table-body');
        if (cachedDocuments.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" class="px-4 py-8 text-center text-slate-400">등록된 문서가 없습니다. 상단에서 파일을 업로드해 보세요.</td></tr>`;
          return;
        }

        tbody.innerHTML = '';
        cachedDocuments.forEach(doc => {
          const score = doc.quality_report ? doc.quality_report.score : 100;
          let badgeClass = "bg-emerald-100 text-emerald-800";
          if (score < 60) badgeClass = "bg-red-100 text-red-800";
          else if (score < 80) badgeClass = "bg-amber-100 text-amber-800";

          const tr = document.createElement('tr');
          tr.className = "hover:bg-slate-50 border-b border-slate-100";
          tr.innerHTML = `
            <td class="px-4 py-3 font-semibold text-slate-900">${escapeHtml(doc.source_name)}</td>
            <td class="px-3 py-3"><span class="px-2 py-0.5 text-xs bg-slate-100 text-slate-600 rounded">${escapeHtml(doc.source_type)}</span></td>
            <td class="px-3 py-3 font-mono font-medium">${doc.chunk_count}개</td>
            <td class="px-3 py-3"><span class="px-2 py-0.5 text-xs rounded-full font-semibold ${badgeClass}">${score}점</span></td>
            <td class="px-4 py-3 text-xs text-slate-400">${doc.created_at ? doc.created_at.slice(0, 16).replace('T', ' ') : '-'}</td>
            <td class="px-4 py-3 text-right space-x-1">
              <button type="button" data-action="view-chunks" data-doc-id="${escapeHtml(String(doc.id))}" data-doc-name="${escapeHtml(doc.source_name)}" class="px-2 py-1 text-xs text-blue-600 hover:bg-blue-50 rounded">청크 보기</button>
              <button type="button" data-action="delete-doc" data-doc-id="${escapeHtml(String(doc.id))}" data-doc-name="${escapeHtml(doc.source_name)}" class="px-2 py-1 text-xs text-red-600 hover:bg-red-50 rounded">삭제</button>
            </td>
          `;
          tbody.appendChild(tr);
        });

      } catch (err) {
        console.error('문서 목록 로드 오류:', err);
      }
    }

    // 청크 조회 모달을 표시함
    async function viewChunks(docId, docName) {
      document.getElementById('modal-doc-title').innerText = docName;
      document.getElementById('modal-doc-subtitle').innerText = `문서 ID: ${docId}`;
      const body = document.getElementById('modal-chunks-body');
      body.innerHTML = `<p class="text-center text-slate-400 py-8">청크를 불러오는 중...</p>`;
      document.getElementById('chunk-modal').classList.remove('hidden');

      try {
        const res = await apiFetch(`/documents/${docId}/chunks`);
        if (!res.ok) throw new Error('청크 조회 실패');
        const chunks = await res.json();

        if (chunks.length === 0) {
          body.innerHTML = `<p class="text-center text-slate-400 py-8">청크가 없습니다.</p>`;
          return;
        }

        body.innerHTML = '';
        chunks.forEach(c => {
          const div = document.createElement('div');
          div.className = "p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-1";
          div.innerHTML = `
            <div class="flex justify-between items-center text-slate-500 font-semibold mb-1">
              <span>청크 #${c.chunk_index}</span>
              <span>글자수: ${c.content.length}자</span>
            </div>
            <pre class="whitespace-pre-wrap font-sans text-slate-800 bg-white p-3 rounded border border-slate-100 text-xs leading-relaxed">${escapeHtml(c.content)}</pre>
          `;
          body.appendChild(div);
        });
      } catch (err) {
        body.innerHTML = `<p class="text-red-500 text-center py-8">${escapeHtml(err.message)}</p>`;
      }
    }

    function closeChunkModal() {
      document.getElementById('chunk-modal').classList.add('hidden');
    }

    // 문서를 삭제함
    async function deleteDoc(docId, docName) {
      if (!confirm(`'${docName}' 문서를 삭제하시겠습니까? 관련 청크와 벡터가 모두 삭제됩니다.`)) return;
      try {
        const res = await apiFetch(`/documents/${docId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('삭제 실패');
        loadDocuments();
      } catch (err) {
        alert('삭제 중 오류 발생: ' + err.message);
      }
    }

    // 파일 업로드를 처리함
    async function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      const statusBox = document.getElementById('upload-status');
      statusBox.classList.remove('hidden');
      statusBox.className = "mt-3 text-xs font-semibold text-blue-600";
      statusBox.innerText = `'${file.name}' 업로드 및 자동 파싱/벡터화 중...`;

      const formData = new FormData();
      formData.append('file', file);
      formData.append('metadata', JSON.stringify({ uploaded_by: 'poc_dashboard' }));

      try {
        const res = await apiFetch('/documents/file', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || '파일 업로드 실패');
        }

        const data = await res.json();
        statusBox.className = "mt-3 text-xs font-semibold text-emerald-600";
        statusBox.innerText = `업로드 완료! (${data.name}, 청크 ${data.chunk_count}개 생성, 품질점수 ${data.quality.score}점)`;
        loadDocuments();
      } catch (err) {
        statusBox.className = "mt-3 text-xs font-semibold text-red-600";
        statusBox.innerText = `업로드 실패: ${err.message}`;
      } finally {
        event.target.value = '';
      }
    }

    // 시스템 통계와 상태를 불러옴
    async function loadStats() {
      try {
        const res = await apiFetch('/stats');
        if (res.ok) {
          const data = await res.json();
          document.getElementById('sys-vllm-model').innerText = data.llm_model || 'rag-vllm-model';
          document.getElementById('sys-embedding').innerText = `${data.embedding_model} (${data.embedding_dimension}차원)`;
        }
      } catch (err) {
        console.error('통계 로드 실패:', err);
      }
    }

    // LMOps 통계 및 최근 트레이스 로그를 불러옴
    async function loadLmopsData() {
      try {
        const sumRes = await apiFetch('/lmops/metrics/summary');
        if (sumRes.ok) {
          const sum = await sumRes.json();
          document.getElementById('lmops-total-queries').innerText = (sum.total_queries || 0).toLocaleString();
          document.getElementById('lmops-avg-latency').innerText = (sum.avg_latency_ms || 0).toFixed(1);
          document.getElementById('lmops-total-tokens').innerText = (sum.total_tokens || 0).toLocaleString();
          document.getElementById('lmops-block-rate').innerText = ((sum.guardrail_block_rate || 0) * 100).toFixed(1);
          document.getElementById('lmops-high-risk').innerText = (sum.high_hallucination_count || 0).toLocaleString();
          document.getElementById('lmops-avg-rating').innerText = (sum.avg_rating || 0).toFixed(1);
          document.getElementById('lmops-feedback-count').innerText = `${sum.feedback_count || 0}건 수집됨`;
        }

        const tracesRes = await apiFetch('/lmops/traces?limit=30');
        const tbody = document.getElementById('lmops-traces-body');
        if (tracesRes.ok) {
          const traces = await tracesRes.json();
          if (traces.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center py-6 text-slate-400">수집된 트레이스가 없습니다.</td></tr>';
          } else {
            tbody.innerHTML = traces.map(t => {
              const actionBadge = t.guardrail_action === 'block'
                ? '<span class="px-2 py-0.5 rounded text-[11px] font-bold bg-red-100 text-red-700">차단(BLOCK)</span>'
                : (t.guardrail_action === 'mask'
                  ? '<span class="px-2 py-0.5 rounded text-[11px] font-medium bg-amber-100 text-amber-700">마스킹</span>'
                  : '<span class="px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-100 text-emerald-700">통과</span>');

              const riskBadge = t.hallucination_risk === 'high'
                ? '<span class="text-red-600 font-bold">주의</span>'
                : (t.hallucination_risk === 'medium'
                  ? '<span class="text-amber-600">보통</span>'
                  : '<span class="text-emerald-600">안전</span>');

              const querySnippet = escapeHtml((t.query_text || '').substring(0, 40));
              const dt = t.created_at ? new Date(t.created_at).toLocaleTimeString() : '-';

              return `
                <tr class="hover:bg-slate-50 transition">
                  <td class="py-2.5 px-3 font-medium text-slate-800" title="${escapeHtml(t.query_text)}">${querySnippet}...</td>
                  <td class="py-2.5 px-3">${actionBadge}</td>
                  <td class="py-2.5 px-3 text-right font-mono">${t.total_latency_ms.toFixed(1)} ms</td>
                  <td class="py-2.5 px-3 text-right font-mono">${t.total_tokens}</td>
                  <td class="py-2.5 px-3">${riskBadge}</td>
                  <td class="py-2.5 px-3 text-slate-500">${dt}</td>
                </tr>
              `;
            }).join('');
          }
        }
      } catch (err) {
        console.error('LMOps 데이터 로드 실패:', err);
      }
    }

    // 고품질 QA를 SFT 학습 데이터셋으로 내보냄
    async function exportCuratedSft() {
      try {
        const res = await apiFetch('/lmops/export-sft', { method: 'POST' });
        if (res.ok) {
          const data = await res.json();
          alert(`SFT 데이터셋 내보내기 완료: 총 ${data.exported_count}건 저장됨 (${data.output_path})`);
        } else {
          alert('SFT 데이터셋 내보내기에 실패했습니다.');
        }
      } catch (err) {
        alert(`오류 발생: ${err.message}`);
      }
    }

    // 가드레일 사전 테스트 실행함
    async function testGuardrail() {
      const text = document.getElementById('guardrail-test-input').value.trim();
      const resBox = document.getElementById('guardrail-test-result');
      if (!text) return;
      resBox.classList.remove('hidden');
      resBox.innerText = '검증 수행 중...';
      try {
        const res = await apiFetch('/guardrails/validate-input', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text, block_on_injection: true, mask_pii: true })
        });
        const data = await res.json();
        resBox.innerHTML = `<pre class="whitespace-pre-wrap">${JSON.stringify(data, null, 2)}</pre>`;
      } catch (err) {
        resBox.innerText = `검증 오류: ${err.message}`;
      }
    }

    // 로컬 RAG 정량 평가기 테스트 실행함
    async function testLocalEval() {
      const query = document.getElementById('eval-test-query').value.trim();
      const answer = document.getElementById('eval-test-answer').value.trim();
      const context = document.getElementById('eval-test-context').value.trim();
      const resBox = document.getElementById('eval-test-result');
      if (!query || !answer) {
        alert('질문과 답변을 모두 입력하세요.');
        return;
      }
      resBox.classList.remove('hidden');
      resBox.innerText = '평가지표 계산 중...';
      try {
        const res = await apiFetch('/eval/rag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            answer: answer,
            contexts: context ? [context] : [],
            engine: 'local'
          })
        });
        const data = await res.json();
        resBox.innerHTML = `
          <div class="font-bold text-slate-800 mb-1">종합 점수: ${data.overall_score} / 100</div>
          <div>충실도(Faithfulness): <strong>${data.faithfulness}</strong></div>
          <div>답변 적합성(Relevance): <strong>${data.answer_relevance}</strong></div>
          <div>문맥 정밀도(Precision): <strong>${data.context_precision}</strong></div>
          <div>환각 위험도: <strong>${data.hallucination_risk}</strong></div>
        `;
      } catch (err) {
        resBox.innerText = `평가 오류: ${err.message}`;
      }
    }

    // 시작 시 서비스 상태를 확인함
    async function checkHealth() {
      try {
        const res = await apiFetch('/health');
        if (res.ok) {
          const data = await res.json();
          if (data.status === 'ok') {
            document.getElementById('status-dot').className = "w-2 h-2 rounded-full bg-emerald-500 animate-pulse";
            document.getElementById('vllm-badge').innerText = "vLLM 11435 : 정상 가동 중";
          } else {
            document.getElementById('status-dot').className = "w-2 h-2 rounded-full bg-amber-500";
            document.getElementById('vllm-badge').innerText = "vLLM : 상태 점검 필요";
          }
        }
      } catch (err) {
        document.getElementById('status-dot').className = "w-2 h-2 rounded-full bg-red-500";
        document.getElementById('vllm-badge').innerText = "vLLM : 연결 끊김";
      }
    }

    function escapeHtml(text) {
      if (!text) return '';
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    // 페이지 로드 시 초기화함
    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('docs-table-body').addEventListener('click', event => {
        const button = event.target.closest('button[data-action]');
        if (!button) return;
        const { action, docId, docName } = button.dataset;
        if (action === 'view-chunks') viewChunks(docId, docName);
        if (action === 'delete-doc') deleteDoc(docId, docName);
      });
      checkHealth();
      loadDocuments();
    });
  </script>
</body>
</html>
"""
