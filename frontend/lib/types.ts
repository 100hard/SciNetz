export interface PaperSummary {
  paper_id: string;
  filename: string;
  status: string;
  uploaded_at: string;
  updated_at: string;
  metadata?: Record<string, unknown> | null;
  errors: string[];
  nodes_written: number;
  edges_written: number;
  co_mention_edges: number;
}

export interface GraphNode {
  id: string;
  label: string;
  type?: string | null;
  aliases: string[];
  times_seen: number;
  section_distribution: Record<string, number>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  relation_verbatim: string;
  confidence: number;
  times_seen: number;
  attributes: Record<string, string>;
  evidence: Record<string, unknown>;
  conflicting: boolean;
  created_at?: string | null;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
}

export interface QAResolvedEntity {
  mention: string;
  candidates: {
    node_id: string;
    name: string;
    selected: boolean;
    similarity: number;
  }[];
}

export interface QAPathEdge {
  src_id: string;
  src_name: string;
  dst_id: string;
  dst_name: string;
  relation: string;
  relation_verbatim: string;
  confidence: number;
  conflicting: boolean;
  attributes: Record<string, string>;
  evidence: {
    doc_id: string;
    element_id: string;
    full_sentence?: string;
  };
}

export interface QAPath {
  edges: QAPathEdge[];
  confidence_product: number;
  section_score: number;
  score: number;
}

export interface QAResponsePayload {
  mode: string;
  summary: string;
  resolved_entities: QAResolvedEntity[];
  paths: QAPath[];
  fallback_edges: QAPathEdge[];
}

export interface UISettings {
  graph_defaults: {
    relations: string[];
    min_confidence: number;
    sections: string[];
    show_co_mentions: boolean;
    layout: string;
  };
}
