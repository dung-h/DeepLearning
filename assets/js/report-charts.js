(function () {
  if (!window.Chart || !window.REPORT_VISUAL_DATA) {
    return;
  }

  var navy = "#14365f";
  var navySoft = "#365d8b";
  var gold = "#b69247";
  var goldSoft = "#d8c08c";

  Chart.defaults.font.family = '"IBM Plex Sans", sans-serif';
  Chart.defaults.color = "#1a2431";
  Chart.defaults.borderColor = "rgba(20,54,95,0.12)";

  function makeDatasets(rawDatasets, isLine) {
    return rawDatasets.map(function (dataset, index) {
      var palette = [
        { border: navy, background: "rgba(20,54,95,0.82)" },
        { border: gold, background: "rgba(182,146,71,0.82)" },
        { border: navySoft, background: "rgba(54,93,139,0.78)" },
        { border: goldSoft, background: "rgba(216,192,140,0.88)" },
      ];
      var color = palette[index % palette.length];
      return {
        label: dataset.label,
        data: dataset.values,
        borderColor: color.border,
        backgroundColor: color.background,
        pointBackgroundColor: color.border,
        pointRadius: isLine ? 3 : 0,
        pointHoverRadius: isLine ? 4 : 0,
        borderWidth: isLine ? 2.2 : 0,
        tension: isLine ? 0.28 : 0,
        fill: false,
        spanGaps: true,
      };
    });
  }

  function buildBarChart(canvasId, config) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !config) {
      return;
    }

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: config.labels,
        datasets: makeDatasets(config.datasets, false),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: {
              usePointStyle: true,
              boxWidth: 10,
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return context.dataset.label + ": " + Number(context.parsed.y).toFixed(4);
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: "#1a2431",
            },
            grid: {
              display: false,
            },
          },
          y: {
            beginAtZero: true,
            suggestedMax: 1,
            ticks: {
              callback: function (value) {
                return Number(value).toFixed(1);
              },
            },
          },
        },
      },
    });
  }

  function buildLineChart(canvasId, config) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !config) {
      return;
    }

    new Chart(canvas, {
      type: "line",
      data: {
        labels: config.labels,
        datasets: makeDatasets(config.datasets, true),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: {
              usePointStyle: true,
              boxWidth: 10,
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return context.dataset.label + ": " + Number(context.parsed.y).toFixed(4);
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: config.xTitle || "Epoch",
            },
            grid: {
              color: "rgba(20,54,95,0.08)",
            },
          },
          y: {
            beginAtZero: true,
            suggestedMax: config.suggestedMax || 1,
            title: {
              display: Boolean(config.yTitle),
              text: config.yTitle || "",
            },
            ticks: {
              callback: function (value) {
                var decimals = typeof config.decimals === "number" ? config.decimals : 2;
                return Number(value).toFixed(decimals);
              },
            },
          },
        },
      },
    });
  }

  buildBarChart("final-text-metrics-chart", window.REPORT_VISUAL_DATA.finalResults.textMetrics);
  buildBarChart("final-mm-metrics-chart", window.REPORT_VISUAL_DATA.finalResults.multimodalMetrics);
  buildLineChart("final-text-loss-chart", window.REPORT_VISUAL_DATA.finalResults.textLoss);
  buildLineChart("final-mm-loss-chart", window.REPORT_VISUAL_DATA.finalResults.multimodalLoss);

  buildBarChart("text-metrics-chart", window.REPORT_VISUAL_DATA.text.metrics);
  buildBarChart("text-per-label-chart", window.REPORT_VISUAL_DATA.text.perLabel);
  buildLineChart("text-history-chart", window.REPORT_VISUAL_DATA.text.history);
  buildLineChart("text-loss-chart", window.REPORT_VISUAL_DATA.text.loss);

  buildBarChart("mm-metrics-chart", window.REPORT_VISUAL_DATA.multimodal.metrics);
  buildBarChart("mm-per-label-chart", window.REPORT_VISUAL_DATA.multimodal.perLabel);
  buildLineChart("mm-history-chart", window.REPORT_VISUAL_DATA.multimodal.history);
  buildLineChart("mm-loss-chart", window.REPORT_VISUAL_DATA.multimodal.loss);
})();
