# Decision Log

## 2024-01-15: Why two-pass span linking?
Context: LLM offsets unreliable
Decision: Separate LLM (relation) from linker (offsets)
Consequences: More complex but 95%+ validation pass rate