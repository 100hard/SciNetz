# Research Paper Knowledge Graph Extraction System
## Production-Ready Implementation Plan

## Progress Snapshot

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Spine & Contracts | ✅ Completed (contracts frozen, config + compose in repo) |
| 1 | Parsing & Metadata | ✅ Completed (Docling pipeline + metadata persisted) |
| 2 | Linguistic Prep | ✅ Completed (spaCy inventory with config flag) |
| 3 | Triplet Extraction | ✅ Completed (two-pass extraction with evidence auditing) |
| 4 | Canonicalization | ✅ Completed (merge map, embeddings, polysemy safeguards) |
| 5 | Graph Writer | ✅ Completed (Neo4j batching, conflict detection, upserts) |
| 6 | Orchestrator w/ Co-mention Fallback | ✅ Completed (idempotent pipeline w/ co-mention fallback) |
| 7 | Graph-first QA | ⏳ Pending |

> Last updated after completing the canonicalization pipeline (Phase 4) and preparing to start the graph writer implementation.

---

## Phase 0 — Spine & Contracts (½ day)

### Goal
Create a stable backbone with versioning support so modules don't break each other.

### Deliverables

**Repo structure:**
- `backend/` (FastAPI, Python 3.11+)
- `frontend/` (Next.js, TypeScript)
- `export/` (HTML template + assets)

**Docker Compose:** api, frontend, neo4j

**Frozen contracts (pydantic models):**

Add these fields to your existing models:
- `ParsedElement`: Add `content_hash` (SHA256 for change detection)
- `Evidence`: Add `full_sentence` (optional, for better UI context)
- `Triplet`: Add `pipeline_version` field (default "1.0.0")
- `Edge`: Add `pipeline_version`, `conflicting` (bool), `created_at` (datetime)
- `Node`: Add `section_distribution` (Dict[str, int]) to track which sections this entity appears in (for polysemy detection)
- **NEW MODEL** `PaperMetadata`: doc_id, title, authors[], year, venue, doi (all optional except doc_id)

**Neo4j bootstrap:**

Add these indexes beyond what you had:
- INDEX on `r.pipeline_version` (for filtering by extraction version)
- INDEX on `r.created_at` (for temporal queries)

**Config file (`config.yaml`):**

Create a central config with these critical parameters:
- `pipeline.version`: "1.0.0"
- `extraction.max_triples_per_chunk_base`: 15
- `extraction.tokens_per_triple`: 60
- `extraction.chunk_size_tokens`: 512
- `extraction.chunk_overlap_tokens`: 50
- `canonicalization.base_threshold`: 0.86
- `canonicalization.polysemy_threshold`: 0.92
- `canonicalization.polysemy_section_diversity`: 3 (if entity appears in ≥3 section types, treat as polysemous)
- `co_mention.enabled`: true
- `co_mention.min_occurrences`: 2 (must co-occur ≥2 times to create edge)
- `co_mention.confidence`: 0.3
- `co_mention.max_distance_chars`: 200
- `qa.entity_match_threshold`: 0.83
- `qa.expand_neighbors`: true
- `qa.neighbor_confidence_threshold`: 0.7
- `qa.max_hops`: 2
- `qa.max_results`: 50
- `export.max_size_mb`: 5
- `export.warn_threshold_mb`: 3
- `export.snippet_truncate_length`: 200

### Tests

- Compose boots, `/health` returns 200
- Golden JSON for 1 tiny sample preserved for CI
- Schema validation: all models parse example JSON payloads
- Config loading test: ensure all required fields present

---

## Phase 1 — Parsing & Metadata (Docling + regex fallback) (1–2 days)

### Goal
PDF → ParsedElement[] with reliable offsets + lightweight metadata extraction.

### Work

**Docling wrapper:**
- Parse PDF to elements
- Apply section normalization heuristics (map "Introduction" → "Intro", "Methodology" → "Methods", etc.)
- Compute content_hash (SHA256) for each element
- Store as `data/parsed/{doc_id}.jsonl`

**Section normalization:**
- Build a mapping dict for common variants (introduction/intro, methodology/methods, conclusion/discussion)
- Use fuzzy string matching (ratio ≥ 0.85) for headers
- If no match, assign `_auto_` as section name

**Lightweight metadata extraction (NO GROBID yet):**
- Extract from first 2 pages using regex patterns:
  - Title: largest font text in first page, or text before "Abstract"
  - Authors: pattern matching for "Name, Name, and Name" or email domains
  - Year: 4-digit pattern 2015-2025 in first 2 pages
  - Venue/conference: look for known conference acronyms (NeurIPS, ICML, ACL, etc.)
  - DOI: regex pattern `10\.\d{4,}/\S+`
- Store PaperMetadata alongside ParsedElement
- This is crude but sufficient for basic filtering; defer GROBID to post-MVP

### Tests

- Non-overlapping character ranges across all elements
- Stable golden outputs for 2 seed PDFs (deterministic parse)
- Malformed PDF returns error object, not crash
- Metadata extraction: at least title + year found on 80% of test papers
- Content hash stability: re-parsing same PDF produces identical hashes

---

## Phase 2 — Linguistic Prep (spaCy minimal; skip A/B) (1 day)

### Goal
Build an entity inventory per chunk to guide extraction.

### Decision: SKIP the A/B test

**Rationale:**
- A/B testing adds complexity without clear measurement at this phase
- You can't measure hallucination accurately until after Phase 3 (span validation)
- The inventory constraint might help or hurt - inconclusive
- **Simpler path:** Build the inventory system but make it OPTIONAL via config flag
- Default to OFF for MVP; enable if initial results show too much noise

### Work

**Entity inventory builder:**
- Use spaCy `en_core_web_sm`: sentence split, noun chunks, NER
- Biomedical detector: if ≥20% of tokens match known biomedical lexicon (MeSH terms, gene symbols), load scispaCy `en_core_sci_md`
- If scispaCy loaded, add abbreviation expansion (e.g., "RL" → "Reinforcement Learning")
- Extract top ≤50 entities per chunk, ranked by:
  - Named entities (PERSON, ORG, PRODUCT, etc.) get priority
  - Noun chunks that appear ≥2 times in chunk
  - Proper nouns (capitalized mid-sentence)
- Deduplicate and filter: remove stopwords, pronouns, single letters

**Config flag:**
```
extraction.use_entity_inventory: false  # Default OFF for MVP
```

### Tests

- Each entity candidate must be an exact substring of the chunk
- No stopwords ("the", "it", "they") or pronouns in inventory
- Biomedical sample (arXiv bio paper) triggers scispaCy branch
- Inventory generation completes in <1s per chunk

---

## Phase 3 — Triplet Extraction (LLM + Two-Pass Span Linking) (2–3 days)

### Goal
Clean S-R-O triples with exact, verifiable evidence spans.

### Critical Design: Two-Pass Architecture

**Pass A: LLM Relation Extraction**

Prompt template (function-call JSON format):

```
SYSTEM: Extract factual relationships from the provided CHUNK.
[OPTIONAL: Focus on entities from CANDIDATE_ENTITIES if provided]

Return JSON array of triples:
{
  "triples": [{
    "subject_text": "exact entity mention",
    "relation_verbatim": "the verb phrase used in text",
    "object_text": "exact entity mention",
    "supportive_sentence": "the sentence that supports this claim",
    "confidence": 0.0-1.0
  }]
}

Rules:
1. Extract only factual, verifiable relationships
2. subject_text and object_text should be specific entities, not pronouns
3. Include the full sentence that best supports each triple
4. confidence: 0.9-1.0 = explicit statement, 0.7-0.89 = strong implication, 0.5-0.69 = weak/indirect
5. Omit triples if uncertain
6. Maximum {max_triples} triples per chunk

Normalize relation_verbatim to ONE of these standard relations:
- is-a, part-of, uses, trained-on, evaluated-on, compared-to, outperforms
- increases, decreases, causes, correlates-with, defined-as

If the relation is passive voice (e.g., "X is used by Y"), convert to active (Y uses X) and swap subject/object.
```

**Pass B: Deterministic Span Linker**

For each triple from Pass A:

1. **Find subject span** in chunk:
   - Try exact substring match
   - Try case-insensitive match
   - Try fuzzy match (Levenshtein distance ≥0.90)
   - If all fail → DROP triple

2. **Find object span** (same cascade)

3. **Find supportive sentence span:**
   - Locate the supportive_sentence in chunk
   - Extract exact (start, end) character offsets
   - If not found → DROP triple

4. **Validation checks:**
   - All offsets within chunk bounds
   - No overlapping subject/object spans in same triple
   - relation_norm ∈ allowed set
   - confidence ∈ [0.0, 1.0]

5. **Directional sanity:**
   - Check for passive indicators: "is {verb}ed by", "was {verb}ed by"
   - If found, flip relation and swap subject/object
   - Update relation_verbatim to active voice

**Caps:**
- `max_triples_per_chunk = min(15, ceil(tokens/60))`
- This scales with content density

### Model Choice & Adapter Pattern

**Implementation status:**
- ✅ OpenAI GPT-4o-mini adapter wired via chat completions API with JSON schema enforcement
- ✅ Two-pass validator emits section-distribution stats for Phase 4 canonicalization

**Start with:**
- GPT-4o-mini (good offset reliability, fast)

**Build an adapter interface:**
- Abstract class `LLMExtractor` with method `extract_triples(chunk, entities) -> List[RawTriple]`
- Implementations: `OpenAIExtractor`, `LocalLLMExtractor`
- Config flag to switch: `extraction.llm_provider: "openai"`
- This allows zero-code swap to Llama-3.1-8B later

### Batching & Rate Limits

- Batch chunks into groups of 5-10 for parallel API calls
- Implement exponential backoff for 429 errors
- Cache LLM responses by `(chunk_content_hash, model_name, prompt_version)`
- Skip reprocessing unchanged chunks

### Tests

**Unit tests:**
- Validator rejects triples with out-of-range spans
- Validator rejects unmapped relation_norm values
- Passive voice detector correctly flips "X is trained on Y" → "Y trained-on X"

**Golden fixtures:**
- Create 10 representative chunks with hand-labeled expected triples
- Mock LLM responses (JSON files)
- Assert exact match on accepted triples after span linking

**Offset reliability benchmark:**
- Run 50 representative chunks through GPT-4o-mini
- Measure: % triples that pass span validation
- Target ≥85% pass rate (tune fuzzy threshold or prompts if below)

**Load test:**
- Process 200 chunks in <5 minutes (with batching)
- Memory stays <2GB throughout

---

## Phase 4 — Canonicalization (E5 + FAISS; conservative + polysemy guard) (2 days)

### Goal
Merge duplicate entities safely while avoiding false merges of polysemous terms.

### Strategy

**Step 1: String normalization (NO singularization)**

For each entity name:
- Lowercase
- Unicode NFKC normalization
- Strip leading/trailing punctuation (keep internal: "GPT-3" stays "gpt-3")
- **DO NOT singularize** technical terms (avoids "Neural Networks" → "Neural Network" error)

**Step 2: Embedding & similarity**

- Use E5-base model for embeddings
- Cache embeddings to `data/embeddings/{entity_name_hash}.npy`
- Build FAISS index (IndexFlatIP for cosine similarity)
- Query threshold: ≥0.86 for initial candidate matches

**Step 3: Polysemy detection & guarding**

Track each entity's `section_distribution` during extraction:
```
{
  "Transformer": {
    "Methods": 15,
    "Results": 8,
    "Intro": 3,
    "Discussion": 2
  }
}
```

**Polysemy rules:**

If an entity appears in ≥3 distinct section types (configurable), it's **potentially polysemous**.

For polysemous entities:
- Require similarity ≥0.92 (stricter than 0.86)
- **Additionally:** Require overlap in section distribution (≥50% overlap in top-2 sections)
- Example: "Transformer" (Methods-heavy) won't merge with "Transformer" (Intro-heavy)

**Alternative polysemy guard (simpler):**

Maintain a manual polysemy blocklist for common ambiguous terms:
```
polysemy_blocklist = [
  "transformer", "regression", "model", "network", "system", 
  "algorithm", "method", "approach", "framework", "attention"
]
```

If entity name (normalized) is in blocklist:
- Require similarity ≥0.94
- Require exact section distribution match (prevents ML Transformer vs Electrical Transformer merges)

**Step 4: Merge execution**

- For each cluster of similar entities (above threshold):
  - Choose canonical name: most frequent name in cluster
  - Create aliases[] list with all variants
  - Assign a stable UUID as canonical `id`
  - Update `times_seen` (sum across cluster)
  - **CRITICAL:** Never rewrite historical node IDs in database
  - Instead: create a merge_map in memory for new writes only

**Step 5: Merge map persistence**

- Store merge decisions in `data/canonicalization/merge_map.json`:
```
{
  "entity_variant_1": "canonical_uuid_abc123",
  "entity_variant_2": "canonical_uuid_abc123",
  ...
}
```
- Use this map during graph writes (Phase 5)
- Allows rollback or manual overrides

### Tests

**Unit tests:**
- Known synonyms merge: "RL", "Reinforcement Learning", "reinforcement learning"
- Clearly different terms don't merge: "BERT" vs "GPT-3"
- Polysemous term case: "Transformer" (ML) vs "Transformer" (electrical) remain separate

**Integration tests:**
- Re-running canonicalizer with same inputs produces identical merge_map (idempotent)
- No new duplicates introduced after merge

**Audit report:**
- Export top 20 clusters by alias count
- Print: canonical name, aliases[], times_seen, section_distribution
- Manual spot-check: ≥18/20 should be correct merges

---

## Phase 5 — Graph Writer (Neo4j + batching) (1 day)

### Goal
Persist nodes/edges with provenance; handle high write volume efficiently.

### Work

**Entity upserter:**
```
upsert_entity(name, type, aliases, section_distribution) -> node_id
```
- Use Neo4j MERGE: match on canonical ID, create if absent
- Update aliases[] and section_distribution on match
- Increment times_seen counter
- Return stable node UUID

**Edge upserter:**
```
upsert_edge(src_id, dst_id, relation_norm, relation_verbatim, evidence, confidence, pipeline_version)
```
- Use MERGE on (src, dst, relation_norm)
- If exists: increment times_seen, update confidence (take max)
- If new: create with all fields
- **CRITICAL:** Evidence field must be populated (validate before write)
- Store attrs dict if present (for future: metric names, values)

**Conflict detection:**

Before writing an edge:
- Query existing edges with same (src, dst, relation_norm)
- If found AND new edge has opposite directionality (detected via relation semantics):
  - Mark BOTH edges as `conflicting=true`
  - Log warning with paper IDs
- Example: "X outperforms Y" (conf=0.9) conflicts with "Y outperforms X" (conf=0.85)

**Batching strategy:**

- Accumulate nodes/edges in memory
- Flush to Neo4j every 200 entities OR 500 edges (whichever comes first)
- Use Neo4j transactions: wrap batches in BEGIN/COMMIT
- On transaction failure: log batch, retry once, then fail-soft (skip batch)

### Tests

**Unit tests:**
- Upserting same node twice produces identical result (idempotent)
- Edge without evidence field raises ValidationError

**Integration tests:**
- Write 1k edges in <10 seconds (with batching)
- No constraint violations (unique node IDs maintained)

**Cypher smoke tests:**
```
MATCH (n:Entity) RETURN count(n)
MATCH ()-[r]->() RETURN count(r)
MATCH ()-[r {conflicting: true}]->() RETURN count(r)
```

---

## Phase 6 — Orchestrator & Fail-Soft Co-Mention (1 day)

### Goal
One endpoint runs the full pipeline safely with idempotence and graceful degradation.

### Work

**Endpoint:** `POST /api/extract/{paper_id}`

**Pipeline steps:**

1. **Parse** (Phase 1): PDF → ParsedElement[] + metadata
2. **Inventory** (Phase 2): elements → entity_inventory (if enabled)
3. **Extract** (Phase 3): chunks → Triplet[] via LLM + span linking
4. **Co-mention fallback** (see below): for failed chunks
5. **Canonicalize** (Phase 4): raw entities → canonical IDs
6. **Graph write** (Phase 5): persist to Neo4j

**Idempotence via content hashing:**

- Before processing each chunk:
  - Check if `content_hash` exists in `processed_chunks` table
  - If yes AND `pipeline_version` unchanged → skip
  - If yes BUT `pipeline_version` newer → reprocess (mark old edges as deprecated)

**Fail-soft strategy: Co-mention edges**

**Problem:** LLM fails (timeout, 429 error, malformed response) for some chunks.

**Solution:** Create low-confidence co-mention edges as fallback.

**Co-mention rules:**

For each chunk where LLM extraction fails:
1. Extract entities using spaCy (from Phase 2 inventory, or fresh NER)
2. Find entity pairs that co-occur in same sentence
3. **Frequency filter:** Only create edge if this (entity_a, entity_b) pair appears in ≥2 sentences across the corpus
4. Create edge:
   - relation_norm: "correlates-with" (neutral default)
   - relation_verbatim: "co-mentioned"
   - confidence: 0.3
   - method: "co-mention"
   - evidence: the sentence span where they co-occur
5. By default, co-mention edges are HIDDEN in UI (user can toggle visibility)

**Why frequency filter?**
- Prevents noise from one-off co-occurrences
- Ensures edge represents a genuine pattern

**Caps:**
- Max 10 co-mention edges per failed chunk
- Max distance: entities within 200 chars in same sentence

### Tests

**E2E integration:**
- Upload 2 PDFs → extract → verify graph non-empty
- Node count > 0, edge count > 0, all edges have evidence

**Idempotence test:**
- Run extraction twice on same paper
- Assert: counts stable, no duplicate nodes/edges

**Fail-soft test:**
- Mock LLM to return 429 error for 20% of chunks
- Pipeline completes successfully
- Co-mention edges created with method="co-mention"
- Default graph view (Phase 8) hides these edges

---

## Phase 7 — Graph-First QA (entity resolution + multi-hop) (2 days)

### Goal
Answer questions using subgraphs + evidence; handle entity ambiguity; enable multi-paper reasoning.

### Endpoint: `POST /api/qa/ask`

**Step 1: Entity resolution from question**

Extract entity mentions from question:
- Use spaCy NER + noun chunks
- For each mention, find matching nodes:
  - Exact name match
  - Alias match
  - Embedding similarity ≥0.83 (from config)
  - Return top-3 candidates per mention

**Ambiguity handling:**
- If multiple candidates, prefer nodes with higher `times_seen`
- If still ambiguous, return ALL candidates and let subgraph ranking decide

**Example:**
```
Question: "Does BERT outperform GPT-2 on GLUE?"
Entities: ["BERT", "GPT-2", "GLUE"]
Resolved nodes:
  - BERT → node_id_123
  - GPT-2 → node_id_456  
  - GLUE → node_id_789, node_id_790 (ambiguous: dataset vs benchmark)
```

**Step 2: Neighborhood expansion**

**Problem with original plan:** Simple 1-hop subgraph misses multi-paper reasoning.

**Solution:** Multi-hop path finding with constraints.

For each pair of resolved entities (A, B):
1. Find all paths from A to B with length ≤ max_hops (default 2)
2. **Filters:**
   - Edge confidence ≥ neighbor_confidence_threshold (0.7)
   - Relation types: user-selected OR default set (uses, trained-on, evaluated-on, compared-to, outperforms, causes)
   - Section preference: Results > Methods > Intro (weighted scoring)
3. Rank paths by:
   - Total path confidence (product of edge confidences)
   - Recency (prefer newer papers via created_at timestamp)
   - Section quality (Results edges > Methods edges)

**Step 3: Subgraph construction**

Collect all nodes/edges from top-K ranked paths (K=10 by default).

**Edge case handling:**
- If zero paths found between any entity pair:
  - Expand to 1-hop neighborhoods of each entity independently
  - Collect edges with confidence ≥0.7
  - This provides context even without direct connection

**Step 4: Evidence collection**

For each edge in subgraph:
- Extract evidence.snippet
- Include full_sentence for context
- Deduplicate snippets from same paper/section

**Step 5: Answer composition**

**Mode A: Direct answer (if clear path exists)**
- Summarize the path(s): "Paper X shows A uses B (confidence 0.92), and Paper Y shows B outperforms C (confidence 0.89)"
- Include inline citations: `<cite doc_id="X" page="5">A uses B</cite>`

**Mode B: Insufficient evidence**
- Return: "Insufficient evidence to answer. Related findings:"
- List top 5 evidence snippets from nearest matches
- Show which entities were resolved and which connections are missing

**Mode C: Conflicting evidence**
- If edges marked `conflicting=true` appear in subgraph:
- Return: "Conflicting evidence found:"
- List both sides with citations and confidence scores

**LLM paraphrasing (optional):**
- Use small LLM (GPT-4o-mini) to rewrite path description
- **CRITICAL CONSTRAINT:** LLM receives ONLY the evidence snippets, no external knowledge
- Prompt: "Rewrite this finding in clearer language. Do not add information. {snippets}"

### Tests

**Unit tests:**
- Entity resolution with typos: "BURT" matches "BERT" with similarity 0.85
- Alias matching: "RL" resolves to "Reinforcement Learning" node

**Integration tests:**
- Question with clear path returns answer with ≥1 citation
- Question with no path returns "Insufficient evidence" + nearest snippets
- Multi-hop case: "Does X cause Z?" finds path X→Y→Z across 2 papers

**Guard tests:**
- When subgraph empty, never fabricate an answer
- LLM paraphrasing doesn't introduce facts not in snippets

---

## Phase 8 — UI (Graph + Evidence + Smart Defaults) (3–4 days)

### Goal
Simple, readable exploration with evidence always one click away.

### Components

**1. Upload & Paper Management**
- Drag-drop upload area
- Paper list with status badges: "Parsing", "Extracting", "Complete", "Failed"
- Show metadata: title, authors, year (from Phase 1)
- Delete/reprocess buttons

**2. Graph View (Cytoscape.js)**

**Default view settings (critical for usability):**
- **Relations shown:** defined-as, uses, trained-on, evaluated-on, compared-to, outperforms
- **Min confidence:** 0.5
- **Sections:** Results, Methods
- **Hide co-mention edges** (method="co-mention")

**Why these defaults?**
- Eliminates noise from low-confidence and exploratory edges
- Focuses on factual, high-value relationships
- Results+Methods sections have the most concrete claims

**Layout:**
- fcose (force-directed with constraints) OR cose-bilkent
- Auto-layout on load, manual drag enabled

**Styling:**
- Color edges by relation_norm (consistent color scheme)
- Node size by degree (larger = more connections)
- Edge thickness by confidence (thicker = higher confidence)
- Highlight conflicting edges in red

**Filters panel:**
- Relation type checkboxes (multi-select)
- Min confidence slider (0.0 - 1.0)
- Section toggles (Intro, Methods, Results, Discussion, All)
- Paper filter (show only selected papers)
- "Show co-mention edges" toggle (default OFF)

**Interactions:**
- Click node: highlight 1-hop neighborhood, show node info panel
- Click edge: open Evidence Panel (see below)
- Hover: tooltip with basic info
- Right-click node: "Expand neighborhood", "Hide node", "Find paths to..."

**3. Evidence Panel (side drawer)**

Triggered by clicking an edge.

**Display:**
- Paper title + authors
- Section name + page number
- Full sentence (evidence.full_sentence)
- Highlighted snippet (evidence.snippet in bold)
- Character range (start-end) for debugging
- Confidence score + method (llm/co-mention)
- "Open source PDF" link if available (future: jump to exact page)

**Multiple edges case:**
- If same entity pair has >1 relation, show all evidence snippets
- Tab interface: "uses (3 papers)" | "outperforms (2 papers)"

**4. QA Panel (bottom drawer or modal)**

- Text input: "Ask a question about the corpus"
- Submit → loading spinner
- Results display:
  - Answer text with inline citations (clickable)
  - "Evidence used" section with snippets
  - If insufficient: "Related findings" with nearest matches
  - If conflicting: Show both sides clearly

**Interactions:**
- Clicking citation opens Evidence Panel with that snippet
- "Show subgraph" button: highlights relevant nodes/edges in graph view

**5. Export button (in header)**

- "Download current view as HTML"
- Calls Phase 9 export with active filter settings
- Shows file size estimate before download

### Tests (Playwright)

**Smoke tests:**
- Upload → extract → graph renders with ≥N nodes
- Filters reduce edge count as expected
- Evidence panel shows exact text matching stored evidence (string equality check)

**Interaction tests:**
- Clicking edge opens Evidence Panel with correct snippet
- Clicking citation in QA opens corresponding evidence
- Default view hides co-mention edges; toggling shows them

**Responsiveness:**
- Graph renders in <3s for 30-paper corpus (200 nodes, 500 edges)
- Filters apply in <500ms

---

## Phase 9 — Interactive HTML Export (1–2 days)

### Goal
One-click self-contained HTML that works offline; prevent size explosion.

### Endpoint

`GET /api/export/html?min_conf=0.5&relations=uses,trained-on&sections=Results,Methods&include_snippets=true&truncate_snippets=false&papers=doc1,doc2`

### Process

**Step 1: Query Neo4j**
- Apply all filters (confidence, relations, sections, papers)
- Return {nodes[], edges[]} with evidence attached

**Step 2: Size check**
```
estimated_size_mb = (len(json.dumps(nodes)) + len(json.dumps(edges))) / 1_000_000

if estimated_size_mb > warn_threshold_mb (3 MB):
  return warning: "Export is large. Options: 
    - Truncate snippets (first 200 chars)
    - Exclude snippets (just IDs)
    - Filter to fewer papers"

if estimated_size_mb > max_size_mb (5 MB):
  return error: "Export too large. Please apply stricter filters or paginate by paper."
```

**Step 3: Snippet handling**

If `truncate_snippets=true`:
- Keep only first 200 chars of evidence.snippet
- Add "... [truncated]" indicator
- Store full snippets in a separate `full_snippets.json` (optional download)

If `include_snippets=false`:
- Store only evidence IDs
- Provide separate endpoint to fetch snippet on demand (requires backend)
- Not truly offline, but keeps file size small

**Step 4: Render HTML template**

Template: `export/scinets_view.html`

**Embedded components:**
- Cytoscape.js library (from CDN or embedded)
- Graph data as inline JSON in `<script>` tag
- Interactive controls:
  - Search box (filter nodes/edges by text)
  - Relation checkboxes (multi-select)
  - Min confidence slider
  - Section toggles
  - Reset layout button
  - Zoom controls
- Evidence side panel (same as Phase 8)
- Legend: relation colors, confidence scale

**Styling:**
- Same visual design as web UI (consistency)
- Color by relation_norm
- Node size by degree
- Edge thickness by confidence

**Buttons:**
- Download PNG/SVG (via Cytoscape.js export)
- Download graph.json (raw data)
- Download graph.graphml (NetworkX-compatible)
- Toggle legend

**Step 5: Additional exports**

Generate these files alongside HTML:
- `graph.json`: raw nodes/edges in portable JSON format
- `graph.graphml`: NetworkX-compatible XML
- `README_export.md`: explains schema, generation date, filters applied, pipeline version
- If bundle requested: zip all files → `export_{timestamp}.zip`

### Tests

**Functional tests:**
- HTML opens in browser without backend (truly offline)
- All interactive controls work (filters, search, layout reset)
- Edge clicks open evidence panel with correct snippet
- Edge/node counts match applied filters

**Format tests:**
- `networkx.read_graphml(graph.graphml)` loads without errors
- JSON schema validation on `graph.json`

**Size tests:**
- 30-paper corpus with snippets: export <3 MB (warn if not)
- 50-paper corpus with truncated snippets: export <5 MB

---

## Phase 10 — Observability & KPIs (1 day)

### Goal
Know when to ship; measure quality continuously.

### Metrics (logged per extraction run)

**Per-phase metrics:**
- Parsed elements count
- Entity candidates count (from inventory)
- Attempted triples count (from LLM)
- Accepted triples count (after span validation)
- Rejected triples count + reasons (out-of-range spans, bad relation, etc.)
- Merged node clusters count
- Final node count, edge count
- Co-mention edges count (fallback edges)

**Performance metrics:**
- Parse time (seconds)
- Extraction time per chunk (p50, p95)
- Canonicalization time
- Graph write time
- End-to-end time per paper

**QA metrics (per query):**
- Entity resolution time
- Subgraph construction time
- Total latency (p50, p95)
- Number of paths found
- Evidence snippets returned

### KPIs (Stop Rules for Shipping)

**These must be GREEN before MVP ships:**

1. **Faithfulness:** ≥90% of QA answers include at least one citation with valid evidence
2. **Hallucination rate:** User-flagged hallucinations <5% (test on 50 sample questions)
3. **Noise control:** Sample 100 edges with conf≥0.5 → ≥85% correct by manual rubric
4. **Duplicate entities:** Audit 100 random nodes → <10% obvious duplicates
5. **QA latency:** p50 ≤ 3 seconds on 20-50 paper corpus
6. **Insight density:** Default graph view reveals ≥3 multi-paper chains (paths spanning ≥2 papers)
7. **Pipeline success rate:** ≥95% of chunks successfully extracted (either LLM or co-mention)
8. **Export usability:** HTML export loads in <5 seconds for 30-paper corpus

### Quality Rubric (for manual edge audit)

**Edge correctness criteria:**

An edge is CORRECT if:
- Subject and object are correctly identified entities
- Relation type accurately represents the relationship stated in evidence
- Evidence snippet actually supports the claimed relationship
- No hallucinated details (dates, metrics, qualifiers not in source)

**Common failure modes to check:**
- Wrong directionality: "A uses B" when text says "B uses A"
- Over-generalization: "A causes B" when text only shows correlation
- Entity boundary errors: "Neural Network Architecture" extracted as two separate entities
- Pronoun resolution failures: "it" incorrectly resolved to wrong entity

### Dashboard & Alerts

**Real-time dashboard (simple HTML page):**
- Current extraction queue status
- Per-paper metrics table (rows = papers, columns = metrics)
- KPI status indicators (green/yellow/red)
- Recent errors log (last 50)

**Alert conditions:**
- Accepted/attempted triple ratio < 0.60 → "Extraction quality degraded"
- QA latency p95 > 10s → "Performance issue"
- Duplicate merge rate > 15% → "Canonicalization too aggressive"

---

## Phase 11 — Reprocessing & Versioning (½ day)

### Goal
Handle pipeline improvements without breaking existing graphs.

### Problem
You fix a bug in extraction logic (Phase 3) or improve canonicalization (Phase 4). Now you want to re-extract all papers with the new version.

### Solution: Version-aware reprocessing

**Version tracking:**
- Every edge has `pipeline_version` field (e.g., "1.0.0", "1.1.0")
- Config file specifies current version

**Reprocessing endpoint:**
`POST /api/reprocess?papers=all&min_version=1.0.0&strategy=deprecate`

**Strategies:**

**1. Deprecate (safe, recommended):**
- Old edges marked as `deprecated=true` (new field on Edge model)
- New edges created with current pipeline_version
- UI can toggle "Show deprecated edges" (default OFF)
- Allows comparison: "What changed between v1.0 and v1.1?"

**2. Replace (destructive):**
- Delete all edges with pipeline_version < current
- Extract from scratch
- Faster, cleaner graph, but loses history

**3. Merge (complex):**
- Keep old edges if new extraction confirms them (confidence within 0.1)
- Add new edges if they're novel
- Mark conflicts if new extraction contradicts old

**Implementation:**
- Check each chunk's content_hash:
  - If hash unchanged AND pipeline_version current → skip
  - If hash unchanged BUT pipeline_version old → reprocess
  - If hash changed → reprocess (paper was updated)

**Migration scripts:**
- `scripts/migrate_v1.0_to_v1.1.py`: Apply specific fixes without full reprocessing
- Example: Fix known directionality errors in "uses" relations

### Tests

**Version migration test:**
- Extract corpus with v1.0.0
- Upgrade pipeline to v1.1.0
- Reprocess with "deprecate" strategy
- Assert: old edges marked deprecated, new edges created, no data loss

**Idempotence test:**
- Reprocess same paper twice with same version
- Assert: no duplicate edges created

---

## Phase 12 — Acceptance Testing & Evaluation (1–2 days)

### Goal
Systematically verify MVP meets all KPIs before launch.

### MVP Acceptance Criteria (from Phase 10)

**Functional requirements:**
1. Upload → extract → graph renders
2. Nodes/edges have click-through evidence (no broken links)
3. QA answers contain citations OR "insufficient evidence" message
4. Filters work correctly (applying filters changes visible nodes/edges)
5. HTML export is interactive and works offline

**Quality requirements:**
1. Duplicate entities <10% on 20-50 paper mixed corpus
2. Default graph view (min_conf=0.5, Results/Methods, key relations) is readable
3. Default graph view reveals ≥3 multi-paper chains

### Evaluation Harness

**Test corpus:**
- 30 papers across 3 domains (10 ML, 10 biomed, 10 climate)
- Mix of arXiv preprints and published papers
- Include 2-3 papers with known errors (malformed PDFs, weird formatting)

**Question bank (50 questions):**

Categories:
- **Definition (15):** "What is BERT?", "Define few-shot learning"
- **Comparison (15):** "How does GPT-3 compare to GPT-2?", "BERT vs ELMo performance"
- **Results (10):** "What accuracy did ResNet achieve on ImageNet?"
- **Limitation (5):** "What are the limitations of Transformer models?"
- **Multi-hop (5):** "Does technique A improve metric B when applied to method C?"

**Metrics to compute:**

1. **Citation rate:** % answers with ≥1 valid citation
2. **Hallucination rate:** % answers with fabricated facts (manual review)
3. **Insufficient evidence rate:** % questions where system correctly says "insufficient"
4. **Multi-hop success:** % multi-hop questions correctly answered

**Human evaluation (2 reviewers):**
- Each reviewer grades 25 answers on 3-point scale:
  - 2 = correct + cited
  - 1 = partially correct OR correct but poorly cited
  - 0 = incorrect OR hallucinated
- Inter-annotator agreement should be ≥0.7 (Cohen's kappa)

**Dedupe audit:**
- Sample 100 random nodes
- Manually identify obvious duplicates (e.g., "BERT" and "bert", "RL" and "Reinforcement Learning")
- Count: should be <10

**Edge quality audit:**
- Sample 100 edges with confidence ≥0.5
- Apply quality rubric (from Phase 10)
- Correct edges: should be ≥85

### Bug Triage Process

For any failure:
1. Categorize: parsing, extraction, canonicalization, QA, UI
2. Severity: blocker (breaks core flow), major (degrades quality), minor (edge case)
3. Assign priority based on: severity × frequency
4. Fix blockers before launch; defer minor issues to post-MVP

### Launch Checklist

Before declaring MVP complete:

- [ ] All 8 KPIs from Phase 10 are GREEN
- [ ] Zero blocker bugs
- [ ] <5 major bugs (documented in issues)
- [ ] README with: setup instructions, architecture diagram, API docs
- [ ] Demo video (3 min): upload → graph → QA → export
- [ ] Deployment guide (Docker Compose on single machine)

**Ship when this checklist is complete.**

---

## Post-MVP Roadmap (Future Phases)

### Phase 13 — GROBID Integration (citations graph)
- Add GROBID Docker service
- Extract structured metadata (title, authors, venue, year, DOI)
- Parse references → create (:Paper)-[:CITES]->(:Paper) edges
- Merge stub papers when target paper is ingested
- Enable citation-based filtering: "Show papers citing X"

### Phase 14 — Advanced Canonicalization
- LLM tie-breaker for ambiguous merges (0.83-0.86 similarity band)
- Entity type constraints: don't merge "Transformer" (ML) with "Transformer" (electrical)
- User feedback loop: "Are these the same entity?" → retrain threshold

### Phase 15 — Scale Optimizations
- Celery/RQ for async extraction (handle 100+ paper queue)
- Redis caching for QA subgraphs (popular queries)
- FAISS GPU index for faster canonicalization
- Neo4j sharding for >100k entities

### Phase 16 — Advanced QA
- Query decomposition: complex questions → sub-questions
- Temporal reasoning: "How has X evolved over time?"
- Comparative analysis: "Summarize all papers about X published after 2020"

### Phase 17 — Collaborative Features
- Multi-user support with authentication
- Shared workspaces (teams share corpus)
- Manual curation: users can merge/split entities, flag bad edges
- Annotation layer: users add notes to nodes/edges

### Phase 18 — Domain Adapters
- Biomedical: PubMed integration, gene/protein ontologies
- Legal: case law citation tracking
- Patents: prior art discovery

---

## Appendix A — Risk Mitigation Summary

### Top 5 Risks & Mitigations

**1. LLM span extraction unreliable**
- **Mitigation:** Two-pass architecture (LLM + deterministic linker)
- **Fallback:** Co-mention edges when LLM fails
- **Testing:** Offset reliability benchmark across models

**2. False merges in canonicalization**
- **Mitigation:** Conservative thresholds (0.86 base, 0.92 for polysemous)
- **Mitigation:** Section distribution checking
- **Mitigation:** Manual audit reports
- **Rollback:** Deprecate bad merges, reprocess with stricter settings

**3. QA entity resolution fails (can't match question entities to graph)**
- **Mitigation:** Multi-strategy matching (exact, alias, embedding)
- **Mitigation:** Expand to 1-hop neighbors as fallback
- **Mitigation:** Return "insufficient evidence" instead of fabricating

**4. Export HTML too large (>10 MB)**
- **Mitigation:** Size warnings at 3 MB, hard limit at 5 MB
- **Mitigation:** Snippet truncation option
- **Mitigation:** Filter-based pagination (export by paper subset)

**5. Pipeline too slow (>30 min per paper)**
- **Mitigation:** Batching LLM calls (5-10 chunks at once)
- **Mitigation:** Caching (embeddings, LLM responses)
- **Mitigation:** Content hash-based skipping (don't reprocess unchanged chunks)
- **Mitigation:** Fail-soft (co-mention) for stuck chunks

---

## Appendix B — Technology Stack Summary

**Backend:**
- Framework: FastAPI (Python 3.11+)
- PDF parsing: Docling
- NLP: spaCy (en_core_web_sm), scispaCy (en_core_sci_md) optional
- Embeddings: E5-base via sentence-transformers
- Similarity: FAISS (IndexFlatIP)
- LLM: OpenAI GPT-4o-mini (adapter pattern ready for future providers)
- Graph DB: Neo4j
- Task queue: (optional) Celery + Redis for post-MVP

**Frontend:**
- Framework: Next.js (TypeScript)
- Graph viz: Cytoscape.js (fcose layout)
- UI components: Tailwind CSS + shadcn/ui
- Testing: Playwright

**Export:**
- Template: HTML + embedded Cytoscape.js
- Formats: JSON, GraphML, PNG/SVG

**DevOps:**
- Containerization: Docker + Docker Compose
- CI/CD: GitHub Actions (run tests on PR)
- Monitoring: Structured JSON logging + simple metrics dashboard

---

## Appendix C — Key Design Decisions & Rationale

**1. Why two-pass span linking instead of pure LLM?**
- LLMs are unreliable with exact character offsets (tested: 60-70% accuracy)
- Deterministic string matching is 99.9% reliable
- Separation allows swapping LLM providers without breaking validation

**2. Why not singularize entity names?**
- Technical terms have semantic differences: "Neural Networks" (field) vs "Neural Network" (single model)
- Stemming/lemmatization breaks acronyms and compound terms
- Better to keep raw form and merge via embeddings

**3. Why co-mention fallback instead of failing silently?**
- Graph completeness matters for navigation (broken chains frustrate users)
- Low-confidence edges still provide value if clearly labeled
- Default-hidden keeps UI clean while preserving data

**4. Why graph-first QA instead of vector search?**
- Structured relationships enable precise reasoning (A→B→C chains)
- Evidence provenance is natural (edges already have citations)
- Vector search is fuzzy; graph queries are auditable

**5. Why interactive HTML export instead of PDF reports?**
- Researchers want to explore, not just read
- Offline HTML is portable (email, USB, archive)
- Self-contained (no server dependency)

**6. Why Neo4j instead of lighter graph DB (e.g., NetworkX in-memory)?**
- Query performance at scale (100k+ entities)
- ACID transactions for concurrent writes
- Cypher query language is expressive
- (NetworkX is fine for <10k entities, but doesn't scale)

**7. Why version tracking on edges instead of graph snapshots?**
- Granular control (reprocess one paper without affecting others)
- Enables diff views ("What changed in v1.1?")
- Smaller storage footprint than full snapshots

---

## Appendix D — Timeline & Resource Estimate

**Assuming:**
- 1 backend engineer (Python, ML experience)
- 1 frontend engineer (TypeScript, React)
- 1 ML/NLP specialist (part-time, for Phases 2-4)

**Realistic timeline:**

| Phase | Duration | Parallelizable? | Dependencies |
|-------|----------|-----------------|--------------|
| 0 - Contracts | 0.5 day | No | None |
| 1 - Parsing | 1-2 days | No | Phase 0 |
| 2 - Linguistic Prep | 1 day | Parallel with 3 | Phase 1 |
| 3 - Extraction | 2-3 days | No | Phase 1 |
| 4 - Canonicalization | 2 days | Parallel with 5 | Phase 3 |
| 5 - Graph Writer | 1 day | Parallel with 4 | Phase 0 |
| 6 - Orchestrator | 1 day | No | Phases 3,4,5 |
| 7 - QA | 2 days | Parallel with 8 | Phases 5,6 |
| 8 - UI | 3-4 days | Parallel with 7 | Phases 5,6 |
| 9 - Export | 1-2 days | No | Phase 8 |
| 10 - Observability | 1 day | Parallel with 9 | All phases |
| 11 - Versioning | 0.5 day | No | Phases 6,5 |
| 12 - Acceptance | 1-2 days | No | All phases |

**Critical path:** 0 → 1 → 3 → 6 → 8 → 9 → 12 = **14-18 days**

**Parallel work can reduce to: 12-15 days** with proper coordination.

**Buffer for unknowns:** +3-4 days

**Total realistic estimate: 15-19 days for MVP**

---

## Appendix E — Success Metrics (6 months post-launch)

Beyond MVP KPIs, measure long-term success:

**Adoption:**
- Weekly active users
- Papers uploaded per week
- Corpus size distribution (median, p95)

**Engagement:**
- QA queries per user session
- Graph interactions (node clicks, filter changes)
- HTML exports downloaded per week

**Quality (ongoing):**
- User-reported bad edges (should decrease over time)
- Manual audit score (should stay ≥85%)
- Reprocessing frequency (indicates pipeline stability)

**Performance:**
- Extraction time per paper (should improve with optimizations)
- QA latency p95 (should stay <5s even as corpus grows)
- Export generation time

**Feature requests (prioritization):**
- Top 5 most-requested features from users
- Friction points (where do users get stuck?)

---

## Final Notes

This plan balances **rigor** (testing, versioning, quality metrics) with **pragmatism** (fail-soft strategies, MVP scope, concrete stop rules).

**Key success factors:**
1. **Phase 3 spike early:** Test LLM span extraction ASAP to validate feasibility
2. **Incremental quality checks:** Don't wait until Phase 12 to evaluate
3. **User feedback loop:** Get 5-10 researchers testing by Phase 9
4. **Resist scope creep:** Ship when KPIs are green, not when "perfect"

**Anti-patterns to avoid:**
- Infinite canonicalization tuning (set threshold, audit, adjust ONCE, move on)
- Over-engineering extraction prompts (good enough > perfect)
- Building features before testing core flow (graph + QA + export are MVP; everything else is nice-to-have)

**When in doubt:**
- Choose simplicity over sophistication
- Choose auditability over automation
- Choose shipping over polishing

**Good luck building!**