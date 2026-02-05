(async function () {
  const summary = await fetch("../ontology/ontology_summary.json").then((r) => r.json());
  const graph = await fetch("../ontology/ontology_graph.json").then((r) => r.json());
  let model = await fetch("../data/model_results.json").then((r) => r.json());
  const realSummary = await fetch("../data/real_summary.json").then((r) => r.json());
  let stats = await fetch("../data/feature_stats.json").then((r) => r.json());

  const $ = (id) => document.getElementById(id);

  $("kpi-classes").textContent = summary.classes.length;
  $("kpi-obj").textContent = summary.object_properties.length;
  $("kpi-data").textContent = summary.datatype_properties.length;
  $("kpi-instances").textContent = summary.instances.length;

  const classList = $("class-list");
  summary.classes.forEach((c) => {
    const span = document.createElement("span");
    span.className = "badge";
    span.textContent = c;
    classList.appendChild(span);
  });

  const propList = $("prop-list");
  [...summary.object_properties, ...summary.datatype_properties].forEach((p) => {
    const span = document.createElement("span");
    span.className = "badge";
    span.textContent = p;
    propList.appendChild(span);
  });

  const grouped = summary.instances.reduce((acc, inst) => {
    acc[inst.class] = acc[inst.class] || [];
    acc[inst.class].push(inst.id);
    return acc;
  }, {});

  const map = $("instance-map");
  Object.keys(grouped)
    .sort()
    .forEach((cls) => {
      const card = document.createElement("div");
      card.className = "map-card";
      const title = document.createElement("h4");
      title.textContent = cls;
      const body = document.createElement("div");
      body.className = "list";
      grouped[cls].forEach((id) => {
        const span = document.createElement("span");
        span.className = "badge";
        span.textContent = id;
        body.appendChild(span);
      });
      card.appendChild(title);
      card.appendChild(body);
      map.appendChild(card);
    });

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
      const td1 = document.createElement("td");
      const td2 = document.createElement("td");
      td1.textContent = inst.id;
      td2.textContent = inst.class;
      tr.appendChild(td1);
      tr.appendChild(td2);
      tableBody.appendChild(tr);
    });

    $("instance-count").textContent = `${filtered.length} shown`;
  };

  renderTable("");
  $("search-input").addEventListener("input", (e) => renderTable(e.target.value));

  // Graph rendering
  const graphDepth = $("graph-depth");
  const graphCount = $("graph-count");
  const graphEl = document.getElementById("graph");

  const renderGraph = (depth) => {
    graphEl.innerHTML = "";

    const width = graphEl.clientWidth || 900;
    const height = 520;

    const svg = d3
      .select(graphEl)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const nodes = graph.nodes.slice(0, 120 * depth);
    const nodeIds = new Set(nodes.map((n) => n.id));
    const edges = graph.edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

    graphCount.textContent = `${nodes.length} nodes`;

    const color = d3.scaleOrdinal()
      .domain(["OilWell", "Equipment", "ProcessVariable", "EventType", "State", "WellComponent", "Sensor"])
      .range(["#f4c430", "#4dd0e1", "#ff8a65", "#7e57c2", "#66bb6a", "#90a4ae", "#26a69a"]);

    const simulation = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id((d) => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-320))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg
      .append("g")
      .attr("stroke", "#2a343e")
      .selectAll("line")
      .data(edges)
      .join("line")
      .attr("stroke-width", 1);

    const node = svg
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", 8)
      .attr("fill", (d) => color(d.type || "Entity"))
      .call(
        d3
          .drag()
          .on("start", (event) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
          })
          .on("drag", (event) => {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
          })
          .on("end", (event) => {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
          })
      );

    const labels = svg
      .append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .attr("font-size", 10)
      .attr("fill", "#cfd8dc")
      .text((d) => d.id);

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      labels.attr("x", (d) => d.x + 10).attr("y", (d) => d.y + 4);
    });
  };

  renderGraph(Number(graphDepth.value));
  graphDepth.addEventListener("input", (e) => renderGraph(Number(e.target.value)));

  // Model metrics
  const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : "n/a");
  $("m-precision").textContent = fmt(model.metrics.precision);
  $("m-recall").textContent = fmt(model.metrics.recall);
  $("m-f1").textContent = fmt(model.metrics.f1);
  $("m-roc").textContent = fmt(model.metrics.roc_auc);
  $("m-pr").textContent = fmt(model.metrics.pr_auc);
  $("m-contam").textContent = fmt(model.contamination);

  // Score histogram chart
  const histCtx = document.getElementById("chart-hist");
  new Chart(histCtx, {
    type: "bar",
    data: {
      labels: model.score_hist.bins.map((b) => b.toFixed(2)),
      datasets: [
        {
          label: "Normal",
          data: model.score_hist.normal,
          backgroundColor: "rgba(77, 208, 225, 0.6)",
        },
        {
          label: "Anomaly",
          data: model.score_hist.anomaly,
          backgroundColor: "rgba(244, 196, 48, 0.6)",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { stacked: true, ticks: { maxTicksLimit: 8 } },
        y: { stacked: true },
      },
      plugins: {
        legend: { labels: { color: "#cfd8dc" } },
      },
    },
  });

  // Timeline chart
  const timeCtx = document.getElementById("chart-timeline");
  new Chart(timeCtx, {
    type: "line",
    data: {
      labels: model.timeline.score.map((_, i) => i),
      datasets: [
        {
          label: "Anomaly Score",
          data: model.timeline.score,
          borderColor: "#4dd0e1",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "True Label",
          data: model.timeline.label.map((v) => v * Math.max(...model.timeline.score)),
          borderColor: "#f4c430",
          borderWidth: 1,
          pointRadius: 0,
          borderDash: [4, 4],
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { display: false },
        y: { ticks: { maxTicksLimit: 6 } },
      },
      plugins: {
        legend: { labels: { color: "#cfd8dc" } },
      },
    },
  });

  // Real data label charts
  const classLabels = Object.keys(realSummary.class_counts);
  const classValues = classLabels.map((k) => realSummary.class_counts[k]);
  const classCtx = document.getElementById("chart-class-counts");
  new Chart(classCtx, {
    type: "bar",
    data: {
      labels: classLabels,
      datasets: [
        {
          label: "Instances",
          data: classValues,
          backgroundColor: "rgba(244, 196, 48, 0.7)",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#cfd8dc" } },
      },
    },
  });

  const stateLabels = Object.keys(realSummary.state_counts);
  const stateValues = stateLabels.map((k) => realSummary.state_counts[k]);
  const stateCtx = document.getElementById("chart-state-counts");
  new Chart(stateCtx, {
    type: "doughnut",
    data: {
      labels: stateLabels,
      datasets: [
        {
          data: stateValues,
          backgroundColor: [
            "rgba(77, 208, 225, 0.7)",
            "rgba(244, 196, 48, 0.7)",
            "rgba(126, 87, 194, 0.7)",
            "rgba(102, 187, 106, 0.7)",
          ],
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#cfd8dc" } },
      },
    },
  });

  // Load real files into dropdown
  fetch("http://localhost:5001/api/list-files")
    .then((r) => r.json())
    .then((data) => {
      const sel = $("file-select");
      data.files.forEach((f) => {
        const opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        sel.appendChild(opt);
      });
    })
    .catch(() => {});

  $("btn-load-file").addEventListener("click", () => {
    const sel = $("file-select");
    if (!sel.value) return;
    $("input-status").textContent = "loading...";
    fetch("http://localhost:5001/api/extract-features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: sel.value }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.features) {
          $("input-json").value = JSON.stringify(data.features, null, 2);
          $("input-status").textContent = "loaded";
        } else {
          $("input-status").textContent = "error";
        }
      })
      .catch(() => {
        $("input-status").textContent = "error";
      });
  });

  // Input evaluation
  let lastContrib = [];

  let featureMean = stats.mean;
  let featureStd = stats.std;

  const deriveFeatures = (obj) => {
    // If object keys already match feature stats, return as-is.
    const keys = Object.keys(obj);
    const statKeys = Object.keys(featureMean);
    if (keys.every((k) => statKeys.includes(k))) {
      return obj;
    }

    // If keys are raw tags with arrays, derive mean/std/min/max
    const derived = {};
    keys.forEach((k) => {
      const val = obj[k];
      if (Array.isArray(val) && val.length > 0) {
        const nums = val.map((v) => Number(v)).filter((v) => Number.isFinite(v));
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
        .map((k) => {
          const z = Math.abs((Number(feats[k]) - featureMean[k]) / (featureStd[k] || 1.0));
          return { feature: k, z };
        })
        .sort((a, b) => b.z - a.z);

      const score = zScores.slice(0, 10).reduce((a, b) => a + b.z, 0) / Math.max(1, Math.min(10, zScores.length));
      const verdict = score > 2.5 ? "Anomalous" : "Normal";

      lastContrib = zScores.slice(0, 8);

      $("model-output").textContent = `Score: ${score.toFixed(2)} | Verdict: ${verdict}`;
      $("input-status").textContent = "evaluated";

      const list = $("contrib-list");
      list.innerHTML = "";
      lastContrib.forEach((c) => {
        const span = document.createElement("span");
        span.className = "badge";
        span.textContent = `${c.feature} (z=${c.z.toFixed(2)})`;
        list.appendChild(span);
      });
    } catch (err) {
      $("input-status").textContent = "invalid JSON";
      $("model-output").textContent = "Could not parse input JSON.";
    }
  };

  $("btn-eval").addEventListener("click", evaluate);

  // Chatbot
  const equipmentByVariable = {};
  graph.edges
    .filter((e) => e.label === "relatedToEquipment")
    .forEach((e) => {
      equipmentByVariable[e.source] = e.target;
    });

  const tagMap = graph.tag_map || {};

  const explain = () => {
    const q = $("chat-question").value.trim();
    if (!q) return;
    const log = $("chat-log");
    const item = document.createElement("div");
    item.className = "chat-message";
    item.textContent = `Q: ${q}\nA: thinking...`;
    log.appendChild(item);

    const payload = {
      question: q,
      score: lastContrib.length ? Number($("model-output").textContent.match(/Score: ([0-9.]+)/)?.[1]) : 0,
      verdict: lastContrib.length ? $("model-output").textContent.split("|")[1]?.trim() : "unknown",
      top_contrib: lastContrib.map((c) => {
        const tag = c.feature.split("_")[0];
        return { tag, z: c.z };
      }),
    };

    fetch("http://localhost:5001/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        item.textContent = `Q: ${q}\nA: ${data.answer}`;
      })
      .catch(() => {
        item.textContent = `Q: ${q}\nA: Could not reach the chatbot server.`;
      });

    $("chat-question").value = "";
  };

  $("chat-send").addEventListener("click", explain);

  // Retrain model
  $("btn-retrain").addEventListener("click", () => {
    $("retrain-status").textContent = "running...";
    fetch("http://localhost:5001/api/retrain", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        if (data.model && data.feature_stats) {
          model = data.model;
          stats = data.feature_stats;
          featureMean = stats.mean;
          featureStd = stats.std;
          $("retrain-status").textContent = "done (reload)";
          window.location.reload();
        } else {
          $("retrain-status").textContent = "error";
        }
      })
      .catch(() => {
        $("retrain-status").textContent = "error";
      });
  });
})();
