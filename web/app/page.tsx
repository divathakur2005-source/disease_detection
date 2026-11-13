"use client";

import { useEffect, useState } from "react";
import type { MLP, ReadyLayer } from "@/lib/model";
import { prepare } from "@/lib/model";
import Assess from "@/components/Assess";
import Results from "@/components/Results";
import Method from "@/components/Method";

type Tab = "assess" | "results" | "method";

export type Loaded = {
  ddx: MLP;
  ddxLayers: ReadyLayer[];
  diab: MLP;
  diabLayers: ReadyLayer[];
  results: any;
};

export default function Page() {
  const [tab, setTab] = useState<Tab>("assess");
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/models/ddxplus.json").then((r) => r.json()),
      fetch("/models/diabetes.json").then((r) => r.json()),
      fetch("/data/results.json").then((r) => r.json()),
    ])
      .then(([ddx, diab, results]) =>
        setData({ ddx, ddxLayers: prepare(ddx), diab, diabLayers: prepare(diab), results })
      )
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <>
      <nav className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "assess"} onClick={() => setTab("assess")}>
          Live assessment
        </button>
        <button role="tab" aria-selected={tab === "results"} onClick={() => setTab("results")}>
          Results
        </button>
        <button role="tab" aria-selected={tab === "method"} onClick={() => setTab("method")}>
          How it works
        </button>
      </nav>

      {error && <div className="card">Could not load the models: {error}</div>}
      {!data && !error && <div className="card">Loading models…</div>}

      {data && tab === "assess" && <Assess data={data} />}
      {data && tab === "results" && <Results results={data.results} />}
      {data && tab === "method" && <Method results={data.results} />}

      <div className="disclaimer">
        <b>Research prototype.</b> This is an academic project demonstrating multimodal machine
        learning. It is clinical decision <i>support</i>, not a diagnosis, and must not be used for
        real medical decisions without qualified medical review. The symptom model is trained on
        DDXPlus, a synthetic dataset generated from a clinical knowledge base.
      </div>
    </>
  );
}
