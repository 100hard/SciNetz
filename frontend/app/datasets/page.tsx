const DatasetsPage = () => {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Datasets</p>
        <h1 className="text-2xl font-semibold text-foreground">Manage knowledge graph datasets</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Track the status of ingested datasets, review metadata coverage, and plan upcoming refresh cycles.
        </p>
      </div>
      <div className="rounded-lg border border-dashed border-primary/40 bg-primary/5 p-8 text-sm text-muted-foreground">
        Dataset management dashboards are coming soon. Configure dataset monitoring and curation workflows here once available.
      </div>
    </div>
  );
};

export default DatasetsPage;
