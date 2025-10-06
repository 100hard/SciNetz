CREATE INDEX relationship_pipeline_version IF NOT EXISTS FOR ()-[r]-() ON (r.pipeline_version);
CREATE INDEX relationship_created_at IF NOT EXISTS FOR ()-[r]-() ON (r.created_at);
