(function () {
  function formatScore(value) {
    return Number(value).toFixed(3);
  }

  function formatPercent(numerator, denominator) {
    if (!denominator) {
      return "0.00%";
    }
    return ((numerator / denominator) * 100).toFixed(2) + "%";
  }

  function formatLabel(label) {
    return label.replaceAll("_", " ");
  }

  function buildSummaryCard(modelName, payload) {
    var exactMatches = payload.total_examples - payload.exact_match_failures;
    var card = document.createElement("article");
    card.className = "error-summary-card";

    var heading = document.createElement("h3");
    heading.textContent = modelName;

    var copy = document.createElement("p");
    copy.className = "body-copy";
    copy.textContent =
      "Tóm tắt lỗi trên tập test, lấy trực tiếp từ checkpoint tốt nhất của mô hình này.";

    var kpis = document.createElement("div");
    kpis.className = "error-kpi-grid";

    [
      {
        value: formatPercent(exactMatches, payload.total_examples),
        label: "Tỷ lệ exact-match đúng",
      },
      {
        value: payload.exact_match_failures.toLocaleString("vi-VN"),
        label: "Mẫu exact-match sai",
      },
      {
        value: payload.total_predicted_positive_labels.toLocaleString("vi-VN"),
        label: "Nhãn dương tính dự đoán",
      },
    ].forEach(function (item) {
      var node = document.createElement("div");
      node.className = "error-kpi";

      var strong = document.createElement("strong");
      strong.textContent = item.value;
      var span = document.createElement("span");
      span.textContent = item.label;

      node.append(strong, span);
      kpis.appendChild(node);
    });

    var hardestTitle = document.createElement("p");
    hardestTitle.className = "error-hardest-title";
    hardestTitle.textContent = "Các nhãn khó nhất";

    var pills = document.createElement("ul");
    pills.className = "pill-list";
    payload.hardest_labels.forEach(function (label) {
      var pill = document.createElement("li");
      pill.className = "pill";
      pill.textContent = formatLabel(label);
      pills.appendChild(pill);
    });

    card.append(heading, copy, kpis, hardestTitle, pills);
    return card;
  }

  function buildDeltaBadge(delta) {
    var badge = document.createElement("span");
    badge.className = "delta-badge " + (delta >= 0 ? "is-positive" : "is-negative");
    badge.textContent = (delta >= 0 ? "+" : "") + formatScore(delta);
    return badge;
  }

  function buildTable(payload) {
    var wrapper = document.createElement("div");
    wrapper.className = "table-wrap";

    var table = document.createElement("table");
    table.className = "results-table";

    table.innerHTML =
      "<thead><tr><th>Nhãn</th><th>Support</th><th>BERT F1</th><th>LSTM F1</th><th>Chênh lệch</th><th>BERT (FN / FP)</th><th>LSTM (FN / FP)</th></tr></thead>";

    var tbody = document.createElement("tbody");
    var lstmByLabel = {};
    payload.lstm.per_label.forEach(function (item) {
      lstmByLabel[item.label] = item;
    });

    payload.bert.per_label
      .slice()
      .sort(function (left, right) {
        return right.f1 - lstmByLabel[right.label].f1 - (left.f1 - lstmByLabel[left.label].f1);
      })
      .forEach(function (bertItem) {
        var lstmItem = lstmByLabel[bertItem.label];
        var row = document.createElement("tr");
        var delta = bertItem.f1 - lstmItem.f1;

        var labelCell = document.createElement("td");
        labelCell.innerHTML = "<strong>" + formatLabel(bertItem.label) + "</strong>";

        var supportCell = document.createElement("td");
        supportCell.textContent = bertItem.support.toLocaleString("vi-VN");

        var bertCell = document.createElement("td");
        bertCell.textContent = formatScore(bertItem.f1);

        var lstmCell = document.createElement("td");
        lstmCell.textContent = formatScore(lstmItem.f1);

        var deltaCell = document.createElement("td");
        deltaCell.appendChild(buildDeltaBadge(delta));

        var bertErrorCell = document.createElement("td");
        bertErrorCell.textContent = bertItem.fn + " / " + bertItem.fp;

        var lstmErrorCell = document.createElement("td");
        lstmErrorCell.textContent = lstmItem.fn + " / " + lstmItem.fp;

        row.append(
          labelCell,
          supportCell,
          bertCell,
          lstmCell,
          deltaCell,
          bertErrorCell,
          lstmErrorCell
        );
        tbody.appendChild(row);
      });

    table.appendChild(tbody);
    wrapper.appendChild(table);
    return wrapper;
  }

  var root = document.getElementById("text-error-analysis-root");
  if (!root) {
    return;
  }

  fetch("../artifacts/text/text_error_analysis.json")
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Failed to load text error analysis");
      }
      return response.json();
    })
    .then(function (payload) {
      root.innerHTML = "";

      var summaryGrid = document.createElement("div");
      summaryGrid.className = "error-summary-grid";
      summaryGrid.append(
        buildSummaryCard("BERT", payload.bert),
        buildSummaryCard("LSTM", payload.lstm)
      );

      var note = document.createElement("p");
      note.className = "results-note";
      note.textContent =
        "Bảng dưới đây được sắp theo mức chênh lệch F1 giữa BERT và LSTM. Những nhãn hiếm như identity_hate và threat là nơi khoảng cách rõ nhất.";

      root.append(summaryGrid, note, buildTable(payload));
    })
    .catch(function () {
      root.innerHTML =
        '<p class="empty-state">Không tải được dữ liệu phân tích lỗi. Các bảng chỉ số và hình tham chiếu phía dưới vẫn khả dụng.</p>';
    });
})();
