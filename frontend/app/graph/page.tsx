import dynamic from "next/dynamic";

const GraphExplorer = dynamic(() => import("../../components/graph-explorer"), {
  ssr: false,
});

const GraphPage = () => {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-primary">Knowledge Graph</p>
        <h1 className="text-2xl font-semibold text-foreground">Explore research connections</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Visualize the relationships between papers and concepts. Expand nodes to progressively load related entities and
          inspect their metadata in the side panel.
        </p>
      </div>

      <GraphExplorer />
    </div>
  );
};

export default GraphPage;
