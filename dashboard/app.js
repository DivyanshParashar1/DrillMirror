const API_BASE = window.DRILLMIRROR_API_BASE || "http://localhost:5001";

(async function () {
  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // ---------- View routing ----------
  const showView = (name) => {
    $$(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === name));
    $$(".side-link").forEach((l) => l.classList.toggle("active", l.dataset.view === name));
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  };

  $$(".side-link").forEach((link) =>
    link.addEventListener("click", (e) => {
      e.preventDefault();
      showView(link.dataset.view);
    })
  );

  // Quick-action buttons in overview
  $$("button[data-goto]").forEach((b) =>
    b.addEventListener("click", () => showView(b.dataset.goto))
  );

  // Initial view from hash
  const initial = (location.hash || "#overview").slice(1);
  if ($$(".view").some((v) => v.dataset.view === initial)) showView(initial);

  // ---------- Theme ----------
  const themeToggle = $("theme-toggle");
  const applyThemeLabel = () => {
    const isLight = document.body.classList.contains("light");
    themeToggle.textContent = isLight ? "☀ Light" : "☾ Dark";
  };
  applyThemeLabel();
  themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("light");
    applyThemeLabel();
  });

  // ---------- Server ping (runs FIRST so pill always updates) ----------
  const setServerStatus = (cls, text) => {
    const dot = $("server-dot");
    const label = $("server-label");
    if (dot) dot.className = `status-dot ${cls}`;
    if (label) label.textContent = text;
  };
  fetch(`${API_BASE}/api/list-files`)
    .then((r) => setServerStatus(r.ok ? "ok" : "err", r.ok ? "server online" : "server error"))
    .catch(() => setServerStatus("err", "server offline"));

  // ---------- Data load ----------
  const fetchJson = async (url, label) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
    return res.json();
  };

  let summary, graph, model, realSummary, stats;
  try {
    [summary, graph, model, realSummary, stats] = await Promise.all([
      fetchJson("../ontology/ontology_summary.json", "ontology_summary"),
      fetchJson("../ontology/ontology_graph.json", "ontology_graph"),
      fetchJson("../data/model_results.json", "model_results"),
      fetchJson("../data/real_summary.json", "real_summary"),
      fetchJson("../data/feature_stats.json", "feature_stats"),
    ]);
  } catch (err) {
    const banner = document.createElement("div");
    banner.className = "load-error";
    banner.innerHTML = `
      <strong>Failed to load dashboard data.</strong><br>
      <code>${err.message}</code><br>
      <span class="muted" style="display:block;margin-top:8px">
        Serve the dashboard from the repo root:<br>
        <code>cd /path/to/DrillMirror &amp;&amp; python3 -m http.server 8000</code><br>
        Then open <code>http://localhost:8000/dashboard/index.html</code>.
      </span>`;
    document.querySelector(".content").prepend(banner);
    return;
  }

  // ---------- Overview KPIs ----------
  const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : "n/a");
  $("kpi-classes").textContent = summary.classes.length;
  $("kpi-obj").textContent = summary.object_properties.length;
  $("kpi-data").textContent = summary.datatype_properties.length;
  $("kpi-instances").textContent = summary.instances.length;
  $("kpi-f1").textContent = fmt(model.metrics.f1);
  $("kpi-pr").textContent = fmt(model.metrics.pr_auc);

  // ---------- Model panel ----------
  $("m-precision").textContent = fmt(model.metrics.precision);
  $("m-recall").textContent = fmt(model.metrics.recall);
  $("m-f1").textContent = fmt(model.metrics.f1);
  $("m-roc").textContent = fmt(model.metrics.roc_auc);
  $("m-pr").textContent = fmt(model.metrics.pr_auc);
  $("m-contam").textContent = fmt(model.contamination);

  const chartColors = () => {
    const s = getComputedStyle(document.body);
    return {
      ink: s.getPropertyValue("--ink").trim() || "#cfd8dc",
      muted: s.getPropertyValue("--muted").trim() || "#8b95a1",
      accent: s.getPropertyValue("--accent").trim() || "#f4c430",
      accent2: s.getPropertyValue("--accent-2").trim() || "#4dd0e1",
    };
  };
  const chartOpts = () => {
    const c = chartColors();
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: c.ink, font: { family: "Inter", size: 12 } } } },
      scales: {
        x: { ticks: { color: c.muted }, grid: { color: "rgba(128,128,128,0.08)" } },
        y: { ticks: { color: c.muted }, grid: { color: "rgba(128,128,128,0.08)" } },
      },
    };
  };

  new Chart($("chart-hist"), {
    type: "bar",
    data: {
      labels: model.score_hist.bins.map((b) => b.toFixed(2)),
      datasets: [
        { label: "Normal", data: model.score_hist.normal, backgroundColor: "rgba(77,208,225,0.65)", borderColor: "rgba(77,208,225,1)", borderWidth: 1 },
        { label: "Anomaly", data: model.score_hist.anomaly, backgroundColor: "rgba(244,196,48,0.65)", borderColor: "rgba(244,196,48,1)", borderWidth: 1 },
      ],
    },
    options: { ...chartOpts(), scales: { x: { stacked: true, ticks: { maxTicksLimit: 8, color: chartColors().muted }, grid: { color: "rgba(128,128,128,0.08)" } }, y: { stacked: true, ticks: { color: chartColors().muted }, grid: { color: "rgba(128,128,128,0.08)" } } } },
  });

  new Chart($("chart-timeline"), {
    type: "line",
    data: {
      labels: model.timeline.score.map((_, i) => i),
      datasets: [
        { label: "Anomaly Score", data: model.timeline.score, borderColor: chartColors().accent2, borderWidth: 2, pointRadius: 0, tension: 0.2 },
        { label: "True Label", data: model.timeline.label.map((v) => v * Math.max(...model.timeline.score)), borderColor: chartColors().accent, borderWidth: 1, pointRadius: 0, borderDash: [4, 4] },
      ],
    },
    options: chartOpts(),
  });

  // Real-data charts
  const classLabels = Object.keys(realSummary.class_counts);
  const classValues = classLabels.map((k) => realSummary.class_counts[k]);
  new Chart($("chart-class-counts"), {
    type: "bar",
    data: { labels: classLabels, datasets: [{ label: "Instances", data: classValues, backgroundColor: "rgba(244,196,48,0.75)", borderColor: chartColors().accent, borderWidth: 1 }] },
    options: chartOpts(),
  });

  const stateLabels = Object.keys(realSummary.state_counts);
  const stateValues = stateLabels.map((k) => realSummary.state_counts[k]);
  new Chart($("chart-state-counts"), {
    type: "doughnut",
    data: {
      labels: stateLabels,
      datasets: [{
        data: stateValues,
        backgroundColor: ["#4dd0e1", "#f4c430", "#a78bfa", "#66bb6a", "#ef5350", "#ff8a65"],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "right", labels: { color: chartColors().ink, font: { family: "Inter", size: 12 } } } },
    },
  });

  // ---------- Events ----------
  const events = [
    { name: "Abrupt Increase of BSW", desc: "A sudden jump in water/sediment percentage. Reduces oil output and raises processing costs." },
    { name: "Spurious Closure of DHSV", desc: "A safety valve closes unexpectedly, cutting flow and triggering immediate production loss." },
    { name: "Severe Slugging", desc: "Strong, repeating flow surges that stress equipment and disrupt production." },
    { name: "Flow Instability", desc: "Irregular swings in pressure/temperature without clear periodicity. Can grow into slugging." },
    { name: "Rapid Productivity Loss", desc: "Production drops quickly as reservoir or flow conditions deteriorate." },
    { name: "Quick Restriction in PCK", desc: "Surface choke valve restricts quickly, often from operations, reducing flow abruptly." },
    { name: "Scaling in PCK", desc: "Mineral buildup in the choke reduces flow over time and may require intervention." },
    { name: "Hydrate in Production Line", desc: "Ice-like hydrates form and block flow, risking long downtime until cleared." },
  ];
  const eventsList = $("events-list");
  events.forEach((e) => {
    const card = document.createElement("div");
    card.className = "event-card";
    card.innerHTML = `<h3>${e.name}</h3><p>${e.desc}</p>`;
    eventsList.appendChild(card);
  });

  // ---------- Classes / properties / glossary ----------
  const classList = $("class-list");
  summary.classes.forEach((c) => {
    const s = document.createElement("span");
    s.className = "chip";
    s.textContent = c;
    classList.appendChild(s);
  });
  const propList = $("prop-list");
  [...summary.object_properties, ...summary.datatype_properties].forEach((p) => {
    const s = document.createElement("span");
    s.className = "chip";
    s.textContent = p;
    propList.appendChild(s);
  });

  const glossary = [
    { term: "OilWell", def: "The whole well system from reservoir to platform." },
    { term: "Equipment", def: "Physical devices used to control or measure flow." },
    { term: "Sensor", def: "Device that measures pressure, temperature, or flow." },
    { term: "PressureSensor", def: "Sensor that measures pressure." },
    { term: "TemperatureSensor", def: "Sensor that measures temperature." },
    { term: "Valve", def: "Device that opens/closes or restricts flow." },
    { term: "Reservoir", def: "Underground zone holding oil and gas." },
    { term: "ProductionTubing", def: "Pipe carrying fluids from reservoir upward." },
    { term: "ProductionLine", def: "Pipe carrying fluids from seabed to platform." },
    { term: "SubseaChristmasTree", def: "Seabed valve/sensor assembly for flow control." },
    { term: "Platform", def: "Surface facility receiving and processing production." },
    { term: "DHSV", def: "Downhole safety valve that shuts the well in emergencies." },
    { term: "PCK", def: "Production choke valve controlling surface flow." },
    { term: "PDG", def: "Downhole pressure gauge." },
    { term: "TPT", def: "Temperature and pressure transducer near the tree." },
    { term: "ProcessVariable", def: "A measured value like pressure, temperature, or flow." },
    { term: "Observation", def: "A single measurement at a time point." },
    { term: "EventType", def: "Category of abnormal behavior (e.g., slugging)." },
    { term: "EventInstance", def: "A specific occurrence of an event in time." },
    { term: "State", def: "Normal, transient, or steady faulty state." },
  ];
  const glossaryEl = $("ontology-glossary");
  glossary.forEach((g) => {
    const item = document.createElement("div");
    item.className = "glossary-item";
    item.innerHTML = `<h4>${g.term}</h4><p>${g.def}</p>`;
    glossaryEl.appendChild(item);
  });

  // ---------- Instance table ----------
  const tableBody = $("instance-table");
  const renderTable = (query) => {
    tableBody.innerHTML = "";
    const filtered = summary.instances.filter((inst) => {
      if (!query) return true;
      const q = query.toLowerCase();
      return inst.id.toLowerCase().includes(q) || inst.class.toLowerCase().includes(q);
    });
    filtered.slice(0, 200).forEach((inst) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${inst.id}</td><td><span class="chip">${inst.class}</span></td>`;
      tableBody.appendChild(tr);
    });
    $("instance-count").textContent = `${filtered.length} shown`;
  };
  renderTable("");
  $("search-input").addEventListener("input", (e) => renderTable(e.target.value));

  // ---------- Graph (D3) ----------
  const graphDepth = $("graph-depth");
  const graphCount = $("graph-count");
  const graphEl = $("graph");

  const renderGraph = (depth) => {
    graphEl.innerHTML = "";
    const width = graphEl.clientWidth || 900;
    const height = 560;
    const svg = d3.select(graphEl).append("svg").attr("width", width).attr("height", height);
    const zoomLayer = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.4, 3]).on("zoom", (e) => zoomLayer.attr("transform", e.transform));
    svg.call(zoom);
    svg.call(zoom.transform, d3.zoomIdentity.scale(0.3 * depth + 0.3));

    const nodes = graph.nodes.slice();
    const edges = graph.edges.slice();
    graphCount.textContent = `${nodes.length} nodes`;

    const color = d3.scaleOrdinal()
      .domain(["OilWell", "Equipment", "ProcessVariable", "EventType", "State", "WellComponent", "Sensor"])
      .range(["#f4c430", "#4dd0e1", "#ff8a65", "#a78bfa", "#66bb6a", "#90a4ae", "#26a69a"]);

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id((d) => d.id).distance(130))
      .force("charge", d3.forceManyBody().strength(-360))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = zoomLayer.append("g").attr("stroke", "rgba(140,150,160,0.35)")
      .selectAll("line").data(edges).join("line").attr("stroke-width", 1);

    const node = zoomLayer.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", 8).attr("fill", (d) => color(d.type || "Entity"))
      .attr("stroke", "rgba(255,255,255,0.15)").attr("stroke-width", 1.5)
      .call(d3.drag()
        .on("start", (e) => { if (!e.active) sim.alphaTarget(0.3).restart(); e.subject.fx = e.subject.x; e.subject.fy = e.subject.y; })
        .on("drag", (e) => { e.subject.fx = e.x; e.subject.fy = e.y; })
        .on("end", (e) => { if (!e.active) sim.alphaTarget(0); e.subject.fx = null; e.subject.fy = null; }));

    const labels = zoomLayer.append("g").selectAll("text").data(nodes).join("text")
      .attr("font-size", 10).attr("fill", chartColors().ink).attr("font-family", "Inter").text((d) => d.id);

    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      labels.attr("x", (d) => d.x + 10).attr("y", (d) => d.y + 4);
    });
  };
  renderGraph(Number(graphDepth.value));
  graphDepth.addEventListener("input", (e) => renderGraph(Number(e.target.value)));

  // ---------- Evaluate tabs ----------
  $$(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      const target = t.dataset.tab;
      $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
      $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === target));
    })
  );

  // ---------- Real-file dropdown ----------
  fetch(`${API_BASE}/api/list-files`)
    .then((r) => r.json())
    .then((data) => {
      const sel = $("file-select");
      (data.files || []).forEach((f) => {
        const opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        sel.appendChild(opt);
      });
    })
    .catch(() => {});

  const setStatus = (id, cls, text) => {
    const el = $(id);
    el.className = `chip ${cls}`;
    el.textContent = text;
  };

  $("btn-load-file").addEventListener("click", () => {
    const sel = $("file-select");
    if (!sel.value) return;
    setStatus("input-status", "loading", "loading…");
    fetch(`${API_BASE}/api/extract-features`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: sel.value }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.features) {
          $("input-json").value = JSON.stringify(data.features, null, 2);
          $$(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === "paste"));
          $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === "paste"));
          setStatus("input-status", "loaded", "loaded");
        } else {
          setStatus("input-status", "err", "error");
        }
      })
      .catch(() => setStatus("input-status", "err", "error"));
  });

  // ---------- Upload parquet ----------
  const uploadDrop = $("upload-drop");
  const uploadInput = $("upload-input");
  const uploadInfo = $("upload-info");

  const doUpload = (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".parquet")) {
      uploadInfo.innerHTML = `<span style="color:var(--error)">Only .parquet files are accepted.</span>`;
      return;
    }
    const fd = new FormData();
    fd.append("file", file);
    setStatus("input-status", "loading", "uploading…");
    uploadInfo.textContent = `Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…`;
    fetch(`${API_BASE}/api/upload-parquet`, { method: "POST", body: fd })
      .then((r) => r.json())
      .then((data) => {
        if (data.features) {
          $("input-json").value = JSON.stringify(data.features, null, 2);
          $$(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === "paste"));
          $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === "paste"));
          setStatus("input-status", "loaded", "loaded");
          uploadInfo.innerHTML = `<span style="color:var(--success)">✓ ${file.name} — extracted ${data.cols} variables.</span>`;
        } else {
          setStatus("input-status", "err", "error");
          uploadInfo.innerHTML = `<span style="color:var(--error)">Upload failed: ${data.error || "unknown"}</span>`;
        }
      })
      .catch((err) => {
        setStatus("input-status", "err", "error");
        uploadInfo.innerHTML = `<span style="color:var(--error)">Upload failed: ${err.message}</span>`;
      });
  };

  uploadDrop.addEventListener("click", () => uploadInput.click());
  $("upload-pick").addEventListener("click", (e) => {
    e.preventDefault();
    uploadInput.click();
  });
  uploadInput.addEventListener("change", (e) => doUpload(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    uploadDrop.addEventListener(ev, (e) => { e.preventDefault(); uploadDrop.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    uploadDrop.addEventListener(ev, (e) => { e.preventDefault(); uploadDrop.classList.remove("dragover"); })
  );
  uploadDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    doUpload(file);
  });

  // ---------- Generate synthetic ----------
  $("btn-generate").addEventListener("click", () => {
    const instances = Number($("gen-instances").value);
    const length = Number($("gen-length").value);
    const filename = $("gen-filename").value.trim();
    if (!filename.endsWith(".csv")) {
      setStatus("gen-status", "err", "filename must end with .csv");
      return;
    }
    setStatus("gen-status", "running", "generating…");
    $("gen-result").innerHTML = `<span class="muted">Running <code>drillmirror.data_pipeline.generate_synthetic</code>…</span>`;
    fetch(`${API_BASE}/api/generate-synthetic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instances, length, filename }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          setStatus("gen-status", "done", "done");
          const kb = (data.size_bytes / 1024).toFixed(1);
          $("gen-result").innerHTML = `
            <div><strong style="color:var(--success)">✓ Generated</strong> ${data.output}</div>
            <div class="muted" style="margin-top:6px">${kb} KB · ${instances} instances × ${length} rows</div>
            ${data.log && data.log.length ? `<div class="muted" style="margin-top:8px;font-family:var(--mono);white-space:pre-wrap">${data.log.join("\n")}</div>` : ""}
          `;
        } else {
          setStatus("gen-status", "err", "error");
          $("gen-result").innerHTML = `<span style="color:var(--error)">Generation failed: ${data.error || "unknown"}</span>`;
        }
      })
      .catch((err) => {
        setStatus("gen-status", "err", "error");
        $("gen-result").innerHTML = `<span style="color:var(--error)">Request failed: ${err.message}</span>`;
      });
  });

  // ---------- Evaluate ----------
  let lastContrib = [];
  let featureMean = stats.mean;
  let featureStd = stats.std;

  const deriveFeatures = (obj) => {
    const keys = Object.keys(obj);
    const statKeys = Object.keys(featureMean);
    if (keys.every((k) => statKeys.includes(k))) return obj;
    const derived = {};
    keys.forEach((k) => {
      const val = obj[k];
      if (Array.isArray(val) && val.length > 0) {
        const nums = val.map(Number).filter(Number.isFinite);
        if (!nums.length) return;
        const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
        const min = Math.min(...nums);
        const max = Math.max(...nums);
        const std = Math.sqrt(nums.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / nums.length);
        derived[`${k}_mean`] = mean;
        derived[`${k}_std`] = std;
        derived[`${k}_min`] = min;
        derived[`${k}_max`] = max;
      }
    });
    return derived;
  };

  const evaluate = () => {
    const text = $("input-json").value.trim();
    if (!text) return;
    try {
      const obj = JSON.parse(text);
      const feats = deriveFeatures(obj);
      const zScores = Object.keys(feats)
        .filter((k) => k in featureMean)
        .map((k) => ({ feature: k, z: Math.abs((Number(feats[k]) - featureMean[k]) / (featureStd[k] || 1.0)) }))
        .sort((a, b) => b.z - a.z);

      const score = zScores.slice(0, 10).reduce((a, b) => a + b.z, 0) / Math.max(1, Math.min(10, zScores.length));
      const verdict = score > 2.5 ? "Anomalous" : "Normal";
      lastContrib = zScores.slice(0, 8);

      $("model-output").innerHTML = `
        <div style="font-size:20px;font-weight:700;letter-spacing:-0.01em">
          <span style="color:${verdict === "Anomalous" ? "var(--error)" : "var(--success)"}">${verdict}</span>
        </div>
        <div class="muted" style="margin-top:6px">Score: ${score.toFixed(2)} · threshold 2.5</div>
      `;
      setStatus("input-status", "evaluated", "evaluated");

      const list = $("contrib-list");
      list.innerHTML = "";
      lastContrib.forEach((c) => {
        const s = document.createElement("span");
        s.className = "chip";
        s.innerHTML = `${c.feature} <strong style="color:var(--accent)">z=${c.z.toFixed(2)}</strong>`;
        list.appendChild(s);
      });
    } catch (err) {
      setStatus("input-status", "err", "invalid JSON");
      $("model-output").textContent = "Could not parse input JSON.";
    }
  };
  $("btn-eval").addEventListener("click", evaluate);

  // ---------- Chatbot ----------
  let chatMode = "engineer";
  $("mode-engineer").addEventListener("click", () => {
    chatMode = "engineer";
    $("mode-engineer").classList.add("active");
    $("mode-manager").classList.remove("active");
  });
  $("mode-manager").addEventListener("click", () => {
    chatMode = "manager";
    $("mode-manager").classList.add("active");
    $("mode-engineer").classList.remove("active");
  });

  const chatLog = $("chat-log");
  const askQuestion = (question) => {
    if (!question) return;
    const empty = chatLog.querySelector(".chat-empty");
    if (empty) empty.remove();

    const item = document.createElement("div");
    item.className = "chat-message";
    item.innerHTML = marked.parse(`**Q:** ${question}\n\n**A:** thinking…`);
    chatLog.appendChild(item);
    chatLog.scrollTop = chatLog.scrollHeight;

    const output = $("model-output").textContent || "";
    const scoreMatch = output.match(/Score:\s*([0-9.]+)/);

    const payload = {
      question,
      score: scoreMatch ? Number(scoreMatch[1]) : 0,
      verdict: output.includes("Anomalous") ? "Anomalous" : output.includes("Normal") ? "Normal" : "unknown",
      mode: chatMode,
      top_contrib: lastContrib.map((c) => ({ tag: c.feature.split("_")[0], z: c.z })),
    };

    fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        item.innerHTML = marked.parse(`**Q:** ${question}\n\n**A:** ${data.answer}`);
        chatLog.scrollTop = chatLog.scrollHeight;
      })
      .catch(() => {
        item.innerHTML = marked.parse(`**Q:** ${question}\n\n**A:** Could not reach the chatbot server.`);
      });
  };

  $("chat-send").addEventListener("click", () => {
    const q = $("chat-question").value.trim();
    if (!q) return;
    askQuestion(q);
    $("chat-question").value = "";
  });
  $("chat-question").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("chat-send").click();
  });
  // Suggested-question chips
  $$(".suggestions .chip").forEach((c) =>
    c.addEventListener("click", () => askQuestion(c.dataset.q))
  );

  // ---------- Reports ----------
  const exportReport = async (type) => {
    const output = $("model-output").textContent || "";
    const scoreMatch = output.match(/Score:\s*([0-9.]+)/);
    const verdict = output.includes("Anomalous") ? "Anomalous" : "Normal";
    const score = scoreMatch ? Number(scoreMatch[1]) : 0;
    const top_contrib = lastContrib.map((c) => ({ tag: c.feature.split("_")[0], z: c.z }));

    const res = await fetch(`${API_BASE}/api/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, verdict, score, top_contrib }),
    });
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${type}_report.pdf`;
    a.click();
    window.URL.revokeObjectURL(url);
  };
  $("btn-report-manager").addEventListener("click", () => exportReport("manager"));
  $("btn-report-engineer").addEventListener("click", () => exportReport("engineer"));

  // ---------- Retrain ----------
  $("btn-retrain").addEventListener("click", () => {
    setStatus("retrain-status", "running", "running…");
    fetch(`${API_BASE}/api/retrain`, { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        if (data.model && data.feature_stats) {
          setStatus("retrain-status", "done", "done (reloading)");
          setTimeout(() => window.location.reload(), 600);
        } else {
          setStatus("retrain-status", "err", "error");
        }
      })
      .catch(() => setStatus("retrain-status", "err", "error"));
  });
})();
