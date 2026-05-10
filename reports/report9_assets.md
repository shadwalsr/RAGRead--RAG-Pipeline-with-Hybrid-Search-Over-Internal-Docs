## 📊 Visual Assets (ASCII Graphs)

### 1. The 429 Rate Limit Bottleneck & Resolution

This diagram visualizes the rate limit issue caused by chaining multiple API
calls, and the production hardening steps taken to fix it.

```text
User Query
    │
    ▼
┌──────────────────┐
│  Embedding API   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Reranking API   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Generation API  │
└────────┬─────────┘
         ▼
┌──────────────────┐      15 req/min limit
│ Verification API │────────────────────────> ( 429 ERROR )
└──────────────────┘                          ( Quota Exc )
                                                    │
                                                    ├──> Fix 1: Rotate API Key
                                                    ├──> Fix 2: time.sleep()
                                                    └──> Fix 3: Graceful Fallbacks
                                                    │
                                                    ▼
                                          [ Pipeline Restored ]
```

### 2. The Hardcoded Model Trap (404 Error)

A sequence diagram showing why hardcoding model versions is dangerous in
production, and how targeting an alias fixes the issue.

```text
Production Code                           Gemini API
      │                                       │
      │        Request: gemini-1.5-flash      │
      ├──────────────────────────────────────>│
      │                                       │
      │             404 NOT_FOUND             │
      │<──────────────────────────────────────┤
      │     (Model not in registry for        │
      │      region/API version)              │
      │                                       │
      │                                       │
      │ Refactor: Use 'gemini-flash-latest'   │
      │                                       │
      │      Request: gemini-flash-latest     │
      ├──────────────────────────────────────>│
      │                                       │
      │                200 OK                 │
      │<──────────────────────────────────────┤
      │       (Generation Successful)         │
      │                                       │
```

### 3. Query Evolution (The "Whyschool Academy" Test)

A visual journey of a single tough query across different development stages.

```text
                     "Whyschool Academy" Query
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      ▼                          ▼                          ▼
[ Report #3 ]              [ Report #6 ]              [ Report #9 ]
 Broke BM25                Triggered 429             Perfect Output
                                                            │
                                               ┌────────────┼────────────┐
                                               ▼            ▼            ▼
                                          Confidence:   Citations:   Completeness:
                                            10/10          100%          High
```

---
