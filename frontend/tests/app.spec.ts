import { test, expect } from "@playwright/test";

const settingsResponse = {
  graph_defaults: {
    relations: ["uses", "outperforms"],
    min_confidence: 0.5,
    sections: ["Results", "Methods"],
    show_co_mentions: false,
    layout: "fcose"
  }
};

const papersResponse = [
  {
    paper_id: "demo-paper",
    filename: "demo.pdf",
    status: "complete",
    uploaded_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    metadata: { title: "Demo Title", year: 2024 },
    errors: [],
    nodes_written: 4,
    edges_written: 3,
    co_mention_edges: 0
  }
];

const graphResponse = {
  nodes: [
    {
      id: "n1",
      label: "Alpha",
      type: "Model",
      aliases: ["Alpha"],
      times_seen: 3,
      section_distribution: { Results: 2 }
    },
    {
      id: "n2",
      label: "Beta",
      type: "Dataset",
      aliases: ["Beta"],
      times_seen: 1,
      section_distribution: { Methods: 1 }
    }
  ],
  edges: [
    {
      id: "edge-1",
      source: "n1",
      target: "n2",
      relation: "uses",
      relation_verbatim: "uses",
      confidence: 0.9,
      times_seen: 1,
      attributes: { method: "llm", section: "Results" },
      evidence: {
        doc_id: "demo-paper",
        element_id: "e1",
        text_span: { start: 0, end: 10 },
        full_sentence: "Alpha uses Beta."
      },
      conflicting: false,
      created_at: "2024-01-01T00:00:00Z"
    }
  ],
  node_count: 2,
  edge_count: 1
};

const qaResponse = {
  mode: "answer",
  summary: "Alpha uses Beta with confidence 0.90.",
  resolved_entities: [
    {
      mention: "Alpha",
      candidates: [{ node_id: "n1", name: "Alpha", selected: true, similarity: 0.9 }]
    },
    {
      mention: "Beta",
      candidates: [{ node_id: "n2", name: "Beta", selected: true, similarity: 0.9 }]
    }
  ],
  paths: [
    {
      edges: [
        {
          src_id: "n1",
          src_name: "Alpha",
          dst_id: "n2",
          dst_name: "Beta",
          relation: "uses",
          relation_verbatim: "uses",
          confidence: 0.9,
          created_at: "2024-01-01T00:00:00Z",
          conflicting: false,
          attributes: { method: "llm" },
          evidence: {
            doc_id: "demo-paper",
            element_id: "e1",
            text_span: { start: 0, end: 10 },
            full_sentence: "Alpha uses Beta."
          }
        }
      ],
      confidence_product: 0.9,
      section_score: 0.8,
      score: 0.9,
      latest_timestamp: "2024-01-01T00:00:00Z"
    }
  ],
  fallback_edges: []
};

test.describe("Graph explorer", () => {
  test("renders defaults and applies filters", async ({ page }) => {
    const graphRequests: URL[] = [];

    await page.route("**/api/ui/settings", (route) => {
      void route.fulfill({ json: settingsResponse });
    });

    await page.route("**/api/ui/papers", (route) => {
      void route.fulfill({ json: papersResponse });
    });

    await page.route("**/api/ui/graph**", (route) => {
      graphRequests.push(new URL(route.request().url()));
      void route.fulfill({ json: graphResponse });
    });

    await page.route("**/api/qa/ask", (route) => {
      void route.fulfill({ json: qaResponse });
    });

    await page.goto("/");

    await expect(page.getByText("demo-paper")).toBeVisible();
    await expect(page.getByText("2 nodes · 1 edge")).toBeVisible();

    const toggle = page.getByLabel("Show co-mention edges");
    await toggle.click();
    await expect.poll(() => graphRequests.length, { message: "graph request not issued" }).toBeGreaterThan(1);
    const lastRequest = graphRequests.at(-1);
    expect(lastRequest?.searchParams.get("include_co_mentions")).toBe("true");

    await page.fill("#qa-input", "Does Alpha use Beta?");
    await page.getByRole("button", { name: "Ask" }).click();
    await expect(page.getByText("Alpha uses Beta.")).toBeVisible();

    await page.getByRole("button", { name: "Highlight" }).click();
    await expect(page.getByText("Evidence")).toBeVisible();
    await expect(page.getByText("Alpha uses Beta.")).toBeVisible();
  });
});
