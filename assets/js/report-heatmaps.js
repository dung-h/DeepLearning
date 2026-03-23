(function () {
  function shorten(label) {
    return label
      .replaceAll("infrastructure_and_utility_damage", "infrastructure / utility damage")
      .replaceAll("rescue_volunteering_or_donation_effort", "rescue / volunteering / donation")
      .replaceAll("other_relevant_information", "other relevant information")
      .replaceAll("affected_individuals", "affected individuals")
      .replaceAll("not_humanitarian", "not humanitarian");
  }

  function textColor(alpha) {
    return alpha > 0.55 ? "#fffdfa" : "#14365f";
  }

  function buildMatrix(container, title, description, labels, matrix) {
    var maxValue = Math.max.apply(null, matrix.flat());
    var card = document.createElement("article");
    card.className = "heatmap-card";

    var heading = document.createElement("h3");
    heading.textContent = title;
    var copy = document.createElement("p");
    copy.textContent = description;

    var shell = document.createElement("div");
    shell.className = "heatmap-shell";
    var table = document.createElement("div");
    table.className = "heatmap-table";

    var header = document.createElement("div");
    header.className = "heatmap-header";
    var corner = document.createElement("div");
    corner.className = "heatmap-corner";
    corner.innerHTML = "<span>Nhãn thật</span><span>Nhãn dự đoán</span>";
    header.appendChild(corner);

    labels.forEach(function (label) {
      var cell = document.createElement("div");
      cell.className = "heatmap-col";
      cell.textContent = shorten(label);
      header.appendChild(cell);
    });

    table.appendChild(header);

    matrix.forEach(function (row, rowIndex) {
      var rowNode = document.createElement("div");
      rowNode.className = "heatmap-row";

      var labelNode = document.createElement("div");
      labelNode.className = "heatmap-row-label";
      labelNode.textContent = shorten(labels[rowIndex]);
      rowNode.appendChild(labelNode);

      row.forEach(function (value, colIndex) {
        var alpha = maxValue === 0 ? 0 : value / maxValue;
        var cell = document.createElement("div");
        cell.className = "heatmap-cell";
        cell.style.background = "rgba(20,54,95," + (0.08 + alpha * 0.82).toFixed(3) + ")";
        cell.style.color = textColor(alpha);
        cell.title = labels[rowIndex] + " -> " + labels[colIndex] + ": " + value;

        var strong = document.createElement("strong");
        strong.textContent = value;
        var meta = document.createElement("span");
        meta.textContent = rowIndex === colIndex ? "đúng lớp" : "nhầm sang lớp khác";
        cell.append(strong, meta);
        rowNode.appendChild(cell);
      });

      table.appendChild(rowNode);
    });

    shell.appendChild(table);
    card.append(heading, copy, shell);
    container.appendChild(card);
  }

  var root = document.getElementById("interactive-confusion-root");
  if (!root) {
    return;
  }

  fetch("../artifacts/multimodal/crisismmd_confusion_matrices.json")
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Failed to load confusion matrices");
      }
      return response.json();
    })
    .then(function (payload) {
      root.innerHTML = "";
      buildMatrix(
        root,
        "Heatmap của CLIP",
        "Heatmap tương tác dựng từ ma trận số đếm thật trên tập test. Di chuột lên từng ô để xem cặp nhãn cụ thể.",
        payload.labels,
        payload.clip
      );
      buildMatrix(
        root,
        "Heatmap của VisualBERT",
        "Heatmap tương tác dựng từ ma trận số đếm thật trên tập test. Cách biểu diễn này giúp đọc nhanh vùng nhầm lẫn hơn ảnh tĩnh.",
        payload.labels,
        payload.visualbert
      );
    })
    .catch(function () {
      root.innerHTML = '<p class="empty-state">Không tải được heatmap tương tác. Ảnh Matplotlib bên dưới vẫn khả dụng.</p>';
    });
})();
