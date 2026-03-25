(function () {
  if (!window.Chart || !window.REPORT_VISUAL_DATA) {
    return;
  }

  var reportData = window.REPORT_VISUAL_DATA || {};
  var finalResults = reportData.finalResults || {};
  var text = reportData.text || {};
  var multimodal = reportData.multimodal || {};

  var palette = [
    {
      border: "#14365f",
      solid: "rgba(20,54,95,0.88)",
      soft: "rgba(20,54,95,0.18)",
      fillStart: "rgba(20,54,95,0.28)",
      fillEnd: "rgba(20,54,95,0.02)",
    },
    {
      border: "#b69247",
      solid: "rgba(182,146,71,0.88)",
      soft: "rgba(182,146,71,0.22)",
      fillStart: "rgba(182,146,71,0.3)",
      fillEnd: "rgba(182,146,71,0.04)",
    },
    {
      border: "#365d8b",
      solid: "rgba(54,93,139,0.84)",
      soft: "rgba(54,93,139,0.2)",
      fillStart: "rgba(54,93,139,0.26)",
      fillEnd: "rgba(54,93,139,0.03)",
    },
    {
      border: "#d8c08c",
      solid: "rgba(216,192,140,0.92)",
      soft: "rgba(216,192,140,0.26)",
      fillStart: "rgba(216,192,140,0.32)",
      fillEnd: "rgba(216,192,140,0.05)",
    },
  ];

  Chart.defaults.font.family = '"IBM Plex Sans", sans-serif';
  Chart.defaults.color = "#1a2431";
  Chart.defaults.borderColor = "rgba(20,54,95,0.12)";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;

  function colorAt(index) {
    return palette[index % palette.length];
  }

  function formatNumber(value, decimals) {
    return Number(value).toFixed(typeof decimals === "number" ? decimals : 4);
  }

  function createGradient(chart, index, horizontal) {
    var chartArea = chart.chartArea;
    var color = colorAt(index);
    if (!chartArea) {
      return color.soft;
    }
    var gradient = horizontal
      ? chart.ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0)
      : chart.ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
    gradient.addColorStop(0, color.fillStart);
    gradient.addColorStop(1, color.fillEnd);
    return gradient;
  }

  function makeDatasets(rawDatasets, kind, options) {
    options = options || {};
    return (rawDatasets || []).map(function (dataset, index) {
      var color = colorAt(index);
      if (kind === "line") {
        return {
          label: dataset.label,
          data: dataset.values,
          borderColor: color.border,
          backgroundColor: function (context) {
            return createGradient(context.chart, index, false);
          },
          pointBackgroundColor: color.border,
          pointBorderColor: "#fffdfa",
          pointBorderWidth: 2,
          pointRadius: 3.5,
          pointHoverRadius: 5,
          borderWidth: 2.4,
          tension: 0.32,
          fill: Boolean(options.fill),
          spanGaps: true,
        };
      }
      return {
        label: dataset.label,
        data: dataset.values,
        borderColor: color.border,
        backgroundColor: function (context) {
          return createGradient(context.chart, index, options.horizontal);
        },
        hoverBackgroundColor: color.soft,
        borderWidth: 1.2,
        borderRadius: 12,
        borderSkipped: false,
        maxBarThickness: options.maxBarThickness || 22,
      };
    });
  }

  function baseTooltip() {
    return {
      backgroundColor: "rgba(15,39,69,0.94)",
      titleColor: "#fffdfa",
      bodyColor: "#eef4ff",
      borderColor: "rgba(255,255,255,0.08)",
      borderWidth: 1,
      cornerRadius: 14,
      padding: 12,
    };
  }

  function buildBarChart(canvasId, config, options) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !config) {
      return;
    }

    options = options || {};

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: config.labels,
        datasets: makeDatasets(config.datasets, "bar", {
          horizontal: Boolean(options.indexAxis === "y"),
          maxBarThickness: options.maxBarThickness,
        }),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: options.indexAxis || "x",
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            position: "top",
          },
          tooltip: Object.assign(baseTooltip(), {
            callbacks: {
              label: function (context) {
                var value = options.indexAxis === "y" ? context.parsed.x : context.parsed.y;
                return context.dataset.label + ": " + formatNumber(value, options.decimals);
              },
            },
          }),
        },
        scales: {
          x: {
            beginAtZero: true,
            suggestedMax: options.indexAxis === "y" ? options.suggestedMax : undefined,
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
            ticks: {
              callback: function (value) {
                if (options.indexAxis === "y") {
                  return formatNumber(value, options.decimals || 2);
                }
                return value;
              },
            },
          },
          y: {
            beginAtZero: options.indexAxis !== "y",
            suggestedMax: options.indexAxis !== "y" ? (options.suggestedMax || 1) : undefined,
            grid: {
              display: options.indexAxis === "y" ? false : true,
              color: "rgba(20,54,95,0.08)",
            },
            ticks: {
              callback: function (value) {
                if (options.indexAxis === "y") {
                  return config.labels[value] || value;
                }
                return formatNumber(value, options.decimals || 2);
              },
            },
          },
        },
      },
    });
  }

  function buildLineChart(canvasId, config, options) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !config) {
      return;
    }

    options = options || {};

    new Chart(canvas, {
      type: "line",
      data: {
        labels: config.labels,
        datasets: makeDatasets(config.datasets, "line", {
          fill: Boolean(options.fill),
        }),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            position: "top",
          },
          tooltip: Object.assign(baseTooltip(), {
            callbacks: {
              label: function (context) {
                return context.dataset.label + ": " + formatNumber(context.parsed.y, options.decimals);
              },
            },
          }),
        },
        scales: {
          x: {
            title: {
              display: true,
              text: options.xTitle || "Epoch",
            },
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
          },
          y: {
            beginAtZero: true,
            suggestedMax: options.suggestedMax || 1,
            title: {
              display: Boolean(options.yTitle),
              text: options.yTitle || "",
            },
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
            ticks: {
              callback: function (value) {
                return formatNumber(value, typeof options.decimals === "number" ? options.decimals : 2);
              },
            },
          },
        },
      },
    });
  }

  function buildMetricSpotlights(containerId, config) {
    var container = document.getElementById(containerId);
    if (!container || !config || !config.labels || !config.datasets || config.datasets.length < 2) {
      return;
    }

    var primary = config.datasets[0];
    var secondary = config.datasets[1];

    container.innerHTML = config.labels.map(function (label, index) {
      var primaryValue = primary.values[index];
      var secondaryValue = secondary.values[index];
      var delta = primaryValue - secondaryValue;
      return [
        '<article class="metric-spotlight-card">',
        '<p class="metric-spotlight-kicker">', label, '</p>',
        '<div class="metric-spotlight-value-row">',
        '<strong>', formatNumber(primaryValue, 4), '</strong>',
        '<span class="metric-spotlight-delta">+', formatNumber(delta, 4), '</span>',
        '</div>',
        '<p class="metric-spotlight-copy"><span>', primary.label, '</span> dẫn trước <span>',
        secondary.label, '</span> trên chỉ số này.</p>',
        '<div class="metric-spotlight-foot">',
        '<span class="metric-pill">', primary.label, '</span>',
        '<span class="metric-vs">vs</span>',
        '<span class="metric-pill is-subtle">', secondary.label, ' ', formatNumber(secondaryValue, 4), '</span>',
        '</div>',
        '</article>',
      ].join("");
    }).join("");
  }

  function perLabelRows(config) {
    if (!config || !config.labels || !config.datasets || config.datasets.length < 2) {
      return [];
    }
    var clip = config.datasets.find(function (item) { return /clip/i.test(item.label); }) || config.datasets[0];
    var visualbert = config.datasets.find(function (item) { return /visualbert/i.test(item.label); }) || config.datasets[1];
    return config.labels.map(function (label, index) {
      return {
        label: label,
        support: (config.support || [])[index] || 0,
        clip: clip.values[index],
        visualbert: visualbert.values[index],
        delta: visualbert.values[index] - clip.values[index],
      };
    });
  }

  function noteForRow(row) {
    if (!row) {
      return "";
    }
    if (row.label === "Well") {
      return "Lớp hiếm nhất và khó nhất; VisualBERT tạo khoảng cách lớn nhất, cho thấy lợi thế rõ rệt trên nhóm mẫu đuôi.";
    }
    if (row.label === "Style") {
      return "Lớp này có ranh giới ngữ nghĩa mờ với Science và Sports, nên rất phù hợp để soi khả năng fusion của mô hình.";
    }
    if (row.label === "Your Money") {
      return "Ảnh và văn bản thường gần với Real Estate, vì vậy đây là nhóm dễ nhầm nhưng có giá trị phân tích cao.";
    }
    if (row.label === "Education") {
      return "Support thấp nhưng VisualBERT vẫn giữ F1 tốt hơn, cho thấy mô hình này ổn định hơn trên lớp đuôi.";
    }
    return "Lớp này giúp minh họa mối quan hệ giữa support, độ khó ngữ nghĩa và mức cải thiện của VisualBERT so với CLIP.";
  }

  function updateClassFocus(containerId, row) {
    var container = document.getElementById(containerId);
    if (!container || !row) {
      return;
    }
    var labelNode = container.querySelector("#mm-class-focus-label");
    var noteNode = container.querySelector("#mm-class-focus-note");
    var supportNode = container.querySelector("#mm-class-focus-support");
    var clipNode = container.querySelector("#mm-class-focus-clip");
    var visualbertNode = container.querySelector("#mm-class-focus-visualbert");
    var deltaNode = container.querySelector("#mm-class-focus-delta");

    if (!labelNode || !noteNode || !supportNode || !clipNode || !visualbertNode || !deltaNode) {
      return;
    }

    labelNode.textContent = row.label;
    noteNode.textContent = noteForRow(row);
    supportNode.textContent = String(row.support);
    clipNode.textContent = formatNumber(row.clip, 4);
    visualbertNode.textContent = formatNumber(row.visualbert, 4);
    deltaNode.textContent = "+" + formatNumber(row.delta, 4);
  }

  function buildGainLandscapeChart(canvasId, config) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
      return;
    }
    var rows = perLabelRows(config).sort(function (a, b) {
      return b.delta - a.delta;
    });

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: rows.map(function (row) { return row.label; }),
        datasets: [
          {
            label: "VisualBERT - CLIP",
            data: rows.map(function (row) { return row.delta; }),
            borderColor: "#14365f",
            backgroundColor: function (context) {
              var chart = context.chart;
              if (!chart.chartArea) {
                return "rgba(20,54,95,0.22)";
              }
              var gradient = chart.ctx.createLinearGradient(chart.chartArea.left, 0, chart.chartArea.right, 0);
              gradient.addColorStop(0, "rgba(182,146,71,0.2)");
              gradient.addColorStop(0.45, "rgba(20,54,95,0.28)");
              gradient.addColorStop(1, "rgba(20,54,95,0.78)");
              return gradient;
            },
            borderRadius: 12,
            borderSkipped: false,
            maxBarThickness: 20,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        plugins: {
          legend: {
            display: false,
          },
          tooltip: Object.assign(baseTooltip(), {
            callbacks: {
              label: function (context) {
                var row = rows[context.dataIndex];
                return [
                  "ΔF1: " + formatNumber(row.delta, 4),
                  "CLIP: " + formatNumber(row.clip, 4),
                  "VisualBERT: " + formatNumber(row.visualbert, 4),
                  "Support: " + row.support,
                ];
              },
            },
          }),
        },
        onHover: function (_, elements) {
          if (elements && elements.length) {
            updateClassFocus("mm-class-focus", rows[elements[0].index]);
          }
        },
        onClick: function (_, elements) {
          if (elements && elements.length) {
            updateClassFocus("mm-class-focus", rows[elements[0].index]);
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            suggestedMax: 0.42,
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
            ticks: {
              callback: function (value) {
                return "+" + formatNumber(value, 2);
              },
            },
            title: {
              display: true,
              text: "Mức tăng F1 của VisualBERT so với CLIP",
            },
          },
          y: {
            grid: {
              display: false,
            },
          },
        },
      },
    });
  }

  function buildSupportGainChart(canvasId, config) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
      return;
    }
    var rows = perLabelRows(config);
    var tailHighlights = (config.tailHighlights || []);

    function bubbleDataset(name, filterFn, color) {
      return {
        label: name,
        data: rows.filter(filterFn).map(function (row) {
          return {
            x: row.support,
            y: row.delta,
            r: Math.max(8, Math.min(20, row.visualbert * 16)),
            meta: row,
          };
        }),
        backgroundColor: color,
        borderColor: color.replace("0.75", "0.95"),
        borderWidth: 1.2,
        hoverBorderWidth: 2,
      };
    }

    new Chart(canvas, {
      type: "bubble",
      data: {
        datasets: [
          bubbleDataset("Lớp spotlight", function (row) {
            return tailHighlights.indexOf(row.label) !== -1;
          }, "rgba(182,146,71,0.82)"),
          bubbleDataset("Các lớp còn lại", function (row) {
            return tailHighlights.indexOf(row.label) === -1;
          }, "rgba(20,54,95,0.72)"),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
          },
          tooltip: Object.assign(baseTooltip(), {
            callbacks: {
              title: function (items) {
                return items[0].raw.meta.label;
              },
              label: function (context) {
                var row = context.raw.meta;
                return [
                  "Support: " + row.support,
                  "ΔF1: +" + formatNumber(row.delta, 4),
                  "CLIP: " + formatNumber(row.clip, 4),
                  "VisualBERT: " + formatNumber(row.visualbert, 4),
                ];
              },
            },
          }),
        },
        onHover: function (_, elements, chart) {
          if (elements && elements.length) {
            var point = chart.data.datasets[elements[0].datasetIndex].data[elements[0].index];
            updateClassFocus("mm-class-focus", point.meta);
          }
        },
        onClick: function (_, elements, chart) {
          if (elements && elements.length) {
            var point = chart.data.datasets[elements[0].datasetIndex].data[elements[0].index];
            updateClassFocus("mm-class-focus", point.meta);
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Support trên tập test",
            },
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
          },
          y: {
            beginAtZero: true,
            suggestedMax: 0.42,
            title: {
              display: true,
              text: "Mức tăng F1",
            },
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
            ticks: {
              callback: function (value) {
                return "+" + formatNumber(value, 2);
              },
            },
          },
        },
      },
    });
  }

  function buildTailSpotlights(containerId, config) {
    var container = document.getElementById(containerId);
    if (!container) {
      return;
    }
    var rows = perLabelRows(config);
    var highlights = (config.tailHighlights || []).map(function (label) {
      return rows.find(function (row) { return row.label === label; });
    }).filter(Boolean);

    if (!highlights.length) {
      highlights = rows.slice().sort(function (a, b) {
        return a.support - b.support;
      }).slice(0, 4);
    }

    var maxSupport = Math.max.apply(null, rows.map(function (row) { return row.support; }));
    var maxDelta = Math.max.apply(null, rows.map(function (row) { return row.delta; }));

    container.innerHTML = highlights.map(function (row) {
      var supportWidth = Math.max(8, (row.support / maxSupport) * 100);
      var deltaWidth = Math.max(10, (row.delta / maxDelta) * 100);
      var note = row.label === "Well"
        ? "Lớp hiếm nhất và khó nhất; VisualBERT tạo khoảng cách lớn nhất."
        : row.label === "Style"
        ? "Không phải lớp hiếm, nhưng ranh giới ngữ nghĩa khá mờ với Science và Sports."
        : row.label === "Your Money"
        ? "Dễ nhầm với Real Estate vì cả ảnh và văn bản cùng thiên về tài sản, nhà ở."
        : "Lớp đuôi với support thấp nhưng VisualBERT vẫn giữ độ ổn định tốt hơn.";
      return [
        '<article class="tail-spotlight-card">',
        '<div class="tail-spotlight-head">',
        '<p class="tail-spotlight-kicker">Spotlight</p>',
        '<h3>', row.label, '</h3>',
        '<span class="tail-delta-chip">ΔF1 +', formatNumber(row.delta, 4), '</span>',
        '</div>',
        '<p class="tail-spotlight-copy">', note, '</p>',
        '<div class="tail-metric-row"><span>Support</span><strong>', row.support, '</strong></div>',
        '<div class="tail-progress"><span style="width:', supportWidth, '%"></span></div>',
        '<div class="tail-metric-row"><span>CLIP F1</span><strong>', formatNumber(row.clip, 4), '</strong></div>',
        '<div class="tail-metric-row"><span>VisualBERT F1</span><strong>', formatNumber(row.visualbert, 4), '</strong></div>',
        '<div class="tail-progress is-delta"><span style="width:', deltaWidth, '%"></span></div>',
        '</article>',
      ].join("");
    }).join("");
  }

  function buildMetricLineChart(canvasId, config) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !config) {
      return;
    }
    new Chart(canvas, {
      type: "line",
      data: {
        labels: config.labels,
        datasets: makeDatasets(config.datasets, "line", { fill: true }),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            position: "top",
          },
          tooltip: Object.assign(baseTooltip(), {
            callbacks: {
              label: function (context) {
                return context.dataset.label + ": " + formatNumber(context.parsed.y, 4);
              },
            },
          }),
        },
        scales: {
          x: {
            grid: {
              display: false,
            },
          },
          y: {
            beginAtZero: true,
            suggestedMax: 1,
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
          },
        },
      },
    });
  }

  buildBarChart("final-text-metrics-chart", finalResults.textMetrics, { suggestedMax: 1, decimals: 4 });
  buildBarChart("final-mm-metrics-chart", finalResults.multimodalMetrics, { suggestedMax: 1, decimals: 4 });
  buildLineChart("final-text-loss-chart", finalResults.textLoss, { suggestedMax: finalResults.textLoss && finalResults.textLoss.suggestedMax, decimals: 3, fill: true });
  buildLineChart("final-mm-loss-chart", finalResults.multimodalLoss, { suggestedMax: finalResults.multimodalLoss && finalResults.multimodalLoss.suggestedMax, decimals: 3, fill: true });

  buildBarChart("text-metrics-chart", text.metrics, { suggestedMax: 1, decimals: 4 });
  buildBarChart("text-per-label-chart", text.perLabel, { suggestedMax: 1, decimals: 4 });
  buildLineChart("text-history-chart", text.history, { suggestedMax: 1, decimals: 4, fill: false });
  buildLineChart("text-loss-chart", text.loss, { suggestedMax: text.loss && text.loss.suggestedMax, decimals: 3, fill: true });

  buildMetricSpotlights("mm-metric-spotlights", multimodal.metrics);
  buildMetricLineChart("mm-metrics-chart", multimodal.metrics);
  buildGainLandscapeChart("mm-gain-chart", multimodal.perLabel);
  buildSupportGainChart("mm-support-chart", multimodal.perLabel);
  buildTailSpotlights("mm-tail-spotlights", multimodal.perLabel);
  var focusRows = perLabelRows(multimodal.perLabel).sort(function (a, b) {
    return b.delta - a.delta;
  });
  if (focusRows.length) {
    updateClassFocus("mm-class-focus", focusRows[0]);
  }
  buildLineChart("mm-history-chart", multimodal.history, { suggestedMax: 1, decimals: 4, fill: false });
  buildLineChart("mm-loss-chart", multimodal.loss, { suggestedMax: multimodal.loss && multimodal.loss.suggestedMax, decimals: 3, fill: true });
})();
