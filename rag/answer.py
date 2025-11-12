# -*- coding: utf-8 -*-
import os, sys, pathlib, re, json
from typing import Dict, Any
from dotenv import load_dotenv
from rag.retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# .env 불러오기
load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

IN_PROMPT   = ROOT / "data" / "index" / "request_prompt.txt"
IN_SUMMARY  = ROOT / "data" / "index" / "input_summary.json"   # ← JSON으로 변경
OUT_ANSWER  = ROOT / "data" / "index" / "answer.txt"; OUT_ANSWER.parent.mkdir(parents=True, exist_ok=True)

# OpenAI 호환 설정
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def build_context_with_required(query: str) -> str:
    """
    벡터 검색 결과에서 기본과실표 / 조정예시 / 법조 / 판례를 모아
    LLM이 그대로 복사해 쓰기 쉽게 섹션으로 구성한다.
    상위 검색된 사고유형ID/이름 섹션도 함께 제공.
    """
    retriever = Retriever()
    results = retriever.search(query, top_k=32)
     #   === 디버그: 검색된 섹션 확인 ===
    # from collections import Counter
    # secs = [r.get("section", "") for r in results]
    # print("\n[DEBUG] 검색된 섹션 개수:", len(secs))
    # print("[DEBUG] 섹션 목록(앞 20개):", secs[:20])
    # print("[DEBUG] 섹션 빈도:", Counter(secs))
    # #각 결과 요약(앞 8개)
    # for i, r in enumerate(results[:8], 1):
    #     print(f"[DEBUG] #{i} section={r.get('section')}  caseID={r.get('사고유형ID')}  caseName={r.get('사고유형명')}")
    base_items, adj_items, law_items, pre_items = [], [], [], []

    # 상위 사고유형 ID/이름 수집
    top_cases = []  # [(id, name)]
    seen_ids = set()
    for r in results:
        cid = (r.get("사고유형ID") or "").strip()
        cname = (r.get("사고유형명") or "").strip()
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            top_cases.append((cid, cname))
        if len(top_cases) >= 8:
            break

    def _split_dash(text: str):
        # "이름 – 본문" / "이름 - 본문" 형태 분리
        parts = re.split(r"\s+[–-]\s+", text, maxsplit=1)
        return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (text.strip(), "")

    def _parse_adj_block(block_text: str):
        out = []
        for ln in block_text.splitlines():
            ln = ln.strip()
            if not ln.startswith("-"): 
                continue
            ln = ln[1:].strip()  # "- " 제거
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 3:
                out.append({"대상": parts[0], "가산사유": parts[1], "조정값": parts[2]})
        return out

    for r in results:
        sec  = (r.get("section", "") or "").strip()
        text = r.get("text", "") or ""
        if not text:
            continue

        # 기본과실표
        if sec.startswith("기본과실비율"):
            base_items.append({
                "A": r.get("A_base"),
                "B": r.get("B_base"),
                "설명": text
            })

        # 조정예시 (이름 변형까지 대응)
        elif any(key in sec for key in ["조정예시", "과실비율조정예시"]):
            adj_items.extend(_parse_adj_block(text))

        # 적용 법조문
        elif any(key in sec for key in ["적용법조항", "법규", "법조문"]):
            name, core = _split_dash(text)
            law_items.append({"조문명": name, "핵심내용": core})

        # 참고 판례
        elif any(key in sec for key in ["참고판례", "판례"]):
            src, gist = _split_dash(text)
            pre_items.append({"출처": src, "판결요지": gist})
    ctx = []

    if top_cases:
        ctx.append("## 검색된 사고유형(상위)")
        for cid, cname in top_cases:
            if cname:
                ctx.append(f"- {cid} – {cname}")
            else:
                ctx.append(f"- {cid}")

    if base_items:
        b = base_items[0]
        ctx.append("## 기본과실표")
        if b.get("A") is not None and b.get("B") is not None:
            ctx.append(f"A={b['A']} / B={b['B']}")
        if b.get("설명"):
            ctx.append(f"설명: {b['설명']}")

    if adj_items:
        ctx.append("## 과실비율조정예시")
        for x in adj_items[:8]:
            ctx.append(f"{x['대상']} | {x['가산사유']} | {x['조정값']}")

    if law_items:
        ctx.append("## 적용 법조문")
        for l in law_items[:4]:
            ctx.append(f"{l.get('조문명','')} – {l.get('핵심내용','')}")

    if pre_items:
        ctx.append("## 참고 판례")
        for p in pre_items[:4]:
            ctx.append(f"{p.get('출처','')} – {p.get('판결요지','')}")

    return "\n".join(ctx)

def call_llm(prompt: str) -> str:
    """
    - 생성: OpenAI 호환 서버(로컬 vLLM/llama.cpp/클라우드 모두 OK)로 호출
    - XAI: 같은 prompt를 HF Transformers로 forward하여 attention/heatmap/CoreRatio 자동 저장
    """
    import os, json, re
    from pathlib import Path
    import requests

    # ===== 1) 생성 (기존 OpenAI Chat Completions 흐름 그대로) =====
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    sys_msg = (
        "너는 한국 교통사고 과실비율 도우미다. 반드시 아래 포맷을 따라라:\n"
        "## 요약 결론\n"
        "- 제안 과실비율: A xx% / B yy%\n\n"
        "## 근거\n"
        "- 기본과실표: 반드시 [검색 컨텍스트]의 '기본과실표'에서 가져온 설명 포함\n"
        "- 조정예시 반영: 반드시 [검색 컨텍스트]의 '과실비율조정예시'에서 가져온 항목 최소 1개 이상 포함\n"
        "- (정규화 필요 시) 총합 100%로 정규화했다는 문장을 포함\n\n"
        "## 적용 법조문\n"
        "- [검색 컨텍스트]의 '적용 법조문'에서 조문명 + 핵심내용을 그대로 포함(의역 금지, 요약 가능)\n\n"
        "## 참고 판례\n"
        "- [검색 컨텍스트]의 '참고 판례'에서 출처 + 판결요지를 그대로 포함(의역 금지, 요약 가능)\n\n"
        "## 입력 요약\n"
        "- video_name: ...\n"
        "- video_date: ...\n"
        "과실비율은 반드시 A+B=100이 되도록 하라."
    )
    payload = {
        "model": OPENAI_MODEL,            # ← vLLM/llama.cpp/Ollama 등 OpenAI-호환 서버면 그대로 동작
        "messages": [
            {"role":"system","content": sys_msg},
            {"role":"user","content": prompt}
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["message"]["content"].strip()

    # ===== 2) XAI 기본 동작: attention/heatmap/CoreRatio 자동 저장 =====
    try:
        import torch, numpy as np, matplotlib
        matplotlib.use("Agg")  # 서버/CLI에서 PNG 저장용
        import matplotlib.pyplot as plt
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import re as _re

        # (필요시 바꿔도 됨) attention 추출용 로컬/허깅페이스 모델
        MODEL_NAME = os.environ.get("XAI_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
        DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(os.environ.get("XAI_DTYPE", "bfloat16"), torch.bfloat16)
        FORCE_CPU = os.environ.get("XAI_FORCE_CPU", "0") == "1"
        MAXLEN = int(os.environ.get("XAI_MAXLEN", "1024"))

        # 산출물 디렉토리
        out_dir = Path("xai_out"); out_dir.mkdir(parents=True, exist_ok=True)

        # 프롬프트 백업(재현성)
        (out_dir/"prompt.txt").write_text(prompt, encoding="utf-8")
        # 응답 백업(참고)
        (out_dir/"answer.txt").write_text(answer, encoding="utf-8")

        # --- HF 로드 & forward (프롬프트만) ---
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=None if FORCE_CPU else ("auto" if torch.cuda.is_available() else None)
        )
        # ★ sdpa → eager 전환: attention 캡처에 필수
        if hasattr(mdl, "set_attn_implementation"):
            mdl.set_attn_implementation("eager")
        # ★ vLLM이 GPU를 점유하므로, 기본은 CPU로 XAI 뽑기(느리지만 확실)
        if FORCE_CPU or (not torch.cuda.is_available()):
            mdl.to("cpu")
        enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True,
        truncation=True,            # ★ 길이 제한 (메모리 절약)
        max_length=MAXLEN).to(mdl.device)

        with torch.inference_mode():
            out = mdl(**{k:enc[k] for k in ["input_ids","attention_mask"]},
                      output_attentions=True, use_cache=False)

        # attentions: Tuple[L](B=1,H,S,S) → [L,H,S,S]
        atts = torch.stack(out.attentions, dim=0).squeeze(1)
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        offsets = enc["offset_mapping"]

        # --- 핵심 토큰 라벨 (정규식; 필요시 수정) ---
        CORE_PATTERNS = [
            r"충돌", r"추돌", r"보행자", r"신호위반",
            r"차로\s*변경", r"정지선", r"속도위반",
            r"우회전", r"좌회전", r"횡단보도"
        ]
        CORE_REGEX = _re.compile("|".join(CORE_PATTERNS))
        spans = [(m.start(), m.end()) for m in CORE_REGEX.finditer(prompt)]

        def _hit(ts:int, te:int) -> bool:
            if te <= ts: return False
            for cs, ce in spans:
                if not (te <= cs or ce <= ts):  # overlap
                    return True
            return False

        core_mask = []
        for s, e in offsets[0].tolist():
            core_mask.append(1 if _hit(s, e) else 0)
        core_mask = np.array(core_mask, dtype=np.int32)

        special_ids = set(tok.all_special_ids)
        ids = enc["input_ids"][0].tolist()
        valid_mask = np.array([0 if t in special_ids else 1 for t in ids], dtype=np.int32)

        # --- CoreRatio 계산 (모든 레이어/헤드 평균) ---
        L, H, S, _ = atts.shape
        key_mask = (valid_mask == 1)
        core_keys = (core_mask == 1) & key_mask
        all_keys  = key_mask
        query_mask = (valid_mask == 1)

        qm = torch.tensor(query_mask, device=atts.device).view(1,1,S,1)
        ck = torch.tensor(core_keys, device=atts.device).view(1,1,1,S)
        ak = torch.tensor(all_keys,  device=atts.device).view(1,1,1,S)

        att_q = atts * qm
        num = (att_q * ck).sum(dim=(-2,-1))  # [L,H]
        den = (att_q * ak).sum(dim=(-2,-1)) + 1e-12
        ratio = (num/den).detach().cpu().numpy()  # [L,H]
        global_mean = float(ratio.mean())
        layer_mean  = ratio.mean(axis=1)
        head_mean   = ratio.mean(axis=0)

        # --- heatmap 저장 (마지막 레이어, head0) ---
        plt.figure(figsize=(8,6))
        mat = atts[-1][0].detach().cpu().numpy()  # [S,S]
        plt.imshow(mat, aspect='auto')
        plt.title("LastLayer-Head0")
        plt.xlabel("Key tokens"); plt.ylabel("Query tokens")
        tick_idx = np.linspace(0, len(tokens)-1, num=min(20, len(tokens))).astype(int)
        plt.xticks(tick_idx, [tokens[i][:8] for i in tick_idx], rotation=90)
        plt.yticks(tick_idx, [tokens[i][:8] for i in tick_idx])
        plt.colorbar(); plt.tight_layout()
        plt.savefig(out_dir/"att_heatmap_last_head0.png", dpi=200)
        plt.close()

        # --- 요약 JSON 저장 ---
        summary = {
            "global_core_ratio": global_mean,
            "layer_mean": layer_mean.tolist(),
            "head_mean":  head_mean.tolist(),
            "num_layers": int(L), "num_heads": int(H), "num_tokens": int(S),
            "core_tokens": int(core_mask.sum()),
            "note": "special tokens excluded; query_scope=all; prompt-only forward"
        }
        (out_dir/"core_ratio_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    except Exception as xe:
        # XAI 실패해도 생성은 살아야 하므로 경고만 찍고 계속
        print("[WARN][XAI] attention 추출 실패:", xe)

    return answer

# ---------- A/B 합 100% 정규화 ----------
_AB_LINE_RE = re.compile(r"(제안\s*과실비율\s*:\s*A\s*)(\d{1,3})(\s*%\s*/\s*B\s*)(\d{1,3})(\s*%)")

def _normalize_ratio_line(ans: str) -> str:
    m = _AB_LINE_RE.search(ans)
    if not m:
        return ans
    a = int(m.group(2)); b = int(m.group(4)); total = a + b
    if total == 100:
        return ans
    new_a = round(a * 100.0 / total)
    new_b = 100 - new_a
    new_line = f"{m.group(1)}{new_a}{m.group(3)}{new_b}{m.group(5)}"
    ans = ans[:m.start()] + new_line + ans[m.end():]
    note = f"- 비율 정규화: 모델 산출 A {a}% / B {b}% (합 {total}%) → 총합 100%로 정규화하여 A {new_a}% / B {new_b}%로 조정"
    ans = _inject_note_under_reason(ans, note)
    return ans

def _inject_note_under_reason(ans: str, note_line: str) -> str:
    header = "## 근거"
    pos = ans.find(header)
    if pos == -1:
        if not ans.endswith("\n"):
            ans += "\n"
        return ans + f"\n{header}\n{note_line}\n"
    line_end = ans.find("\n", pos + len(header))
    if line_end == -1:
        line_end = len(ans)
    insert_at = line_end + 1
    return ans[:insert_at] + note_line + "\n" + ans[insert_at:]

def _load_summary_fields() -> Dict[str, str]:
    """input_summary.json에서 video_name/date를 안전하게 읽어온다."""
    if IN_SUMMARY.exists():
        try:
            obj = json.loads(IN_SUMMARY.read_text(encoding="utf-8"))
            return {
                "video_name": obj.get("video_name", "미정"),
                "video_date": obj.get("video_date", "미정"),
            }
        except Exception:
            pass
    # 백업: 없거나 파싱 실패하면 미정
    return {"video_name": "미정", "video_date": "미정"}

# -------------------- 메인 --------------------
def main():
    if not IN_PROMPT.exists():
        print(f"[ERR] prompt not found: {IN_PROMPT}")
        sys.exit(1)

    prompt_core = IN_PROMPT.read_text(encoding="utf-8")

    # --- input_summary.json 직접 로드 ---
    video_name, video_date = "미정", "미정"
    if IN_SUMMARY.exists():
        try:
            obj = json.loads(IN_SUMMARY.read_text(encoding="utf-8"))
            video_name = obj.get("video_name", "미정")
            video_date = obj.get("video_date", "미정")
        except Exception as e:
            print("[WARN] input_summary.json 파싱 실패:", e)

    # 필수 섹션 확보
    context = build_context_with_required(prompt_core)

    # 최종 프롬프트
    prompt = (
        f"{prompt_core}\n\n"
        f"[입력 요약]\n"
        f"- video_name: {video_name}\n"
        f"- video_date: {video_date}\n\n"
        f"[검색 컨텍스트]\n{context}"
    )

    if not OPENAI_API_KEY:
        print("="*70)
        print("[NO OPENAI_API_KEY] 프롬프트만 출력합니다. (모델 호출 없음)")
        print("="*70)
        print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
        print("="*70)
        print("(OPENAI_API_KEY/OPENAI_MODEL/OPENAI_BASE_URL을 .env에 설정하세요)")
        return

    print("[LLM] 호출 시작...")
    ans = call_llm(prompt)

    # A/B 합 100% 정규화 & 근거 삽입
    ans = _normalize_ratio_line(ans)

    OUT_ANSWER.write_text(ans, encoding="utf-8")

    print("="*70)
    print("[FINAL ANSWER]")
    print(ans)
    print("="*70)
    print(f"[OK] saved → {OUT_ANSWER}")

if __name__ == "__main__":
    main()
