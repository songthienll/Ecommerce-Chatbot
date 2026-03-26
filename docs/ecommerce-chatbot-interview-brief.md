# Ecommerce-Chatbot Interview Brief

## 1) Co con viec gi can lam khong?
Con, neu muon vao interview mode “safe + thuyet phuc”:

- Add tests thuc su cho `pipeline.py` + `retriever.py` (hien pytest collect 0 tests).
- Remove CI bypass:
  - `pytest --tb=short -q || true`
  - `npm run typecheck || true`
- Chuẩn hóa docs trong `docs/` (hien README da tot, nhung docs folder cua repo can update de khop implementation moi).
- Docker local issue la may ban (WSL timeout), khong phai code; CI da pass docker build.

## 2) Project nay giai bai toan gi?
Vietnamese ecommerce assistant cho Tiki:

- User hoi bang tieng Viet.
- He thong tim ngu canh san pham tu vector DB.
- LLM tao cau tra loi + tra ve danh sach sources de hien thi card san pham.

Muc tieu: tang chat luong tu van san pham, giam hallucination, van giu latency nhanh.

## 3) Stack chinh

- Frontend: React 18 + TypeScript + Vite
- Backend: FastAPI (Python)
- Vector DB: Qdrant (migrate tu Chroma)
- Embedding: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- LLM: Groq `llama-3.3-70b-versatile`
- Rerank: Cohere (optional, fail-safe skip)
- Deploy/runtime: Docker Compose + GitHub Actions CI

## 4) Luong xu ly (end-to-end)

1. Frontend goi `POST /api/chat`.
2. Backend `pipeline.answer()`:
   - detect off-topic / product-intent
   - retrieve chunks tu Qdrant qua `retriever.retrieve()`
   - relevance gate + domain gate
   - neu pass: build `product_context`
   - generate answer qua Groq
3. API tra `{answer, sources}`.

Endpoints:
- `POST /api/chat`
- `GET /api/products`
- `GET /api/health`

## 5) Architecture decisions quan trong de noi trong phong van

### A. Qdrant migration
- Ly do: production-fit hon local Chroma cho data lớn, API/ops ro rang hon.
- Luu y critical: Qdrant score semantics “higher is better” (khac Chroma distance).
- Ban da fix fusion scoring theo semantics dung.

### B. Hybrid retrieval
Retriever ket hop:
- vector score
- keyword/price score
- brand boost
- electronics-term boost
- noise penalty

=> giam truong hop lexical bait (vd: query tai nghe ma hit do choi).

### C. Relevance gate + intent gate
Trong `pipeline.py`:
- off-topic greetings bi suppress sources
- product-intent query khong bi over-block
- category conflict / hard constraints check
- fallback out-of-scope cho laptop RTX khi dataset khong co

=> answer “an toan”, han che source rac.

### D. Optional reranking
Cohere loi/het key -> skip re-rank, he thong van chay.
Fail-open co kiem soat.

## 6) Ket qua evaluation hien tai

Tu eval set 25 questions (backend/data/eval/results.json):
- hit@1 = 0.48 | hit@3 = 0.52 | hit@5 = 0.52
- mrr = 0.50 | precision@1 = 0.48 | precision@5 = 0.224
- rouge_l = 0.277 | faithfulness = 3.28 | relevance = 2.04

New eval set 30 queries + runner: `backend/tests/query_eval_set.json` + `query-eval-runner.py`
(Chua chay duoc vi Qdrant collection trong — can ingest/migrate data truoc.)

Manual test results:
- `xin chao` => sources = 0 (dung)
- `laptop gaming rtx 4050 duoi 20 trieu` => sources = 0 + fallback out-of-scope
- query tai nghe Sony kho => sources co relevance tot hon truoc

CI run moi nhat: success (bao gom docker build jobs tren GitHub runner)

## 7) Diem yeu / tech debt (can noi thang)

- Test coverage thuc te = 0 (gap lon nhat).
- CI dang “cho qua” test/typecheck bang `|| true`.
- Dataset khong bao phu het danh muc (vd laptop RTX), can acceptance strategy ro rang.
- Docker local bi khoa boi WSL timeout tren may dev (khong phai project bug).

## 8) Neu bi hoi “ban se lam gi tiep?”

Roadmap ngan gon:
1. Viet test cho retrieval/gating regressions.
2. Bo `|| true` trong CI, bat buoc pass that.
3. Add eval set (20-50 queries) + scorecard precision@k theo domain.
4. Data enrichment cho phone/laptop categories.
5. Add observability metrics: retrieval hit quality, fallback rate, source/domain mismatch.

## 9) Cau tra loi mau 30s

"Day la chatbot RAG tieng Viet cho ecommerce. Em migrate tu Chroma sang Qdrant, sua lai scoring do Qdrant co semantics nguoc Chroma, them hybrid retrieval (vector + keyword + brand boost + noise penalty), va lam relevance gate de chan source rac. Em cung them fallback out-of-scope cho cac case dataset khong cover nhu laptop RTX 4050. CI da pass tren GitHub Actions, con viec uu tien tiep theo la test coverage va bo cac `|| true` trong CI de nang quality gate len production level."

## 10) Unresolved questions

- Team co chap nhan fail-open cho reranker trong production lau dai khong?
- Muc SLA latency/quality target cu the cho `/api/chat` la bao nhieu?
- Co plan bo sung data laptop/phone de giam fallback rate khong?
