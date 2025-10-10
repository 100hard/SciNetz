const SettingsPage = () => {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Settings</p>
        <h1 className="text-2xl font-semibold text-foreground">Configure SciNets</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Adjust ingestion controls, API credentials, and workspace preferences to tailor SciNets for your research team.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-foreground">Workspace preferences</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Centralize global options like theme, notification cadence, and collaboration defaults. Detailed controls are coming soon.
          </p>
        </div>
        <div className="rounded-lg border border-dashed border-primary/40 bg-primary/5 p-6 text-sm text-muted-foreground">
          API credential management will appear here. Rotate tokens, review permissions, and audit integrations in an upcoming update.
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
