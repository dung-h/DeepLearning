(function () {
  function createElement(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (typeof text === "string") {
      node.textContent = text;
    }
    return node;
  }

  function renderCards(root, items) {
    root.innerHTML = "";

    if (!items.length) {
      root.appendChild(
        createElement(
          "div",
          "sample-empty",
          "Không có mẫu nào khớp với bộ lọc hiện tại."
        )
      );
      return;
    }

    items.forEach(function (item) {
      var card = createElement("article", "sample-card");

      var image = document.createElement("img");
      image.src = item.image;
      image.alt = "Ảnh mẫu lớp " + item.label;
      card.appendChild(image);

      var meta = createElement("div", "sample-meta");
      meta.appendChild(createElement("span", "", item.event));
      meta.appendChild(createElement("span", "", item.split));
      meta.appendChild(createElement("span", "", item.label));
      card.appendChild(meta);

      card.appendChild(createElement("h3", "", item.title));
      card.appendChild(createElement("p", "", item.text));

      var actions = createElement("div", "branch-link-row");
      actions.style.marginTop = "0.85rem";

      var openImage = createElement("a", "button-link button-secondary", "Mở ảnh mẫu");
      openImage.href = item.image;
      openImage.target = "_blank";
      openImage.rel = "noreferrer";
      actions.appendChild(openImage);

      card.appendChild(actions);
      root.appendChild(card);
    });
  }

  function renderFilters(root, labels, activeKey, onSelect) {
    root.innerHTML = "";

    labels.forEach(function (item) {
      var button = createElement("button", "sample-filter", item.label);
      button.type = "button";
      if (item.key === activeKey) {
        button.classList.add("is-active");
      }
      button.addEventListener("click", function () {
        onSelect(item.key);
      });
      root.appendChild(button);
    });
  }

  function initPreview() {
    var data = window.MULTIMODAL_PREVIEW_DATA || [];
    var cardRoot = document.getElementById("multimodal-preview-root");
    var filterRoot = document.getElementById("multimodal-preview-filters");
    var countRoot = document.getElementById("multimodal-preview-count");

    if (!cardRoot || !filterRoot || !countRoot || !data.length) {
      return;
    }

    var labelMap = [];
    var seen = {};

    data.forEach(function (item) {
      if (!seen[item.label]) {
        seen[item.label] = true;
        labelMap.push({ key: item.label, label: item.labelTitle });
      }
    });

    var filters = [{ key: "all", label: "Tất cả" }].concat(labelMap);
    var active = "all";

    function refresh() {
      var items =
        active === "all"
          ? data
          : data.filter(function (item) {
              return item.label === active;
            });

      renderFilters(filterRoot, filters, active, function (nextKey) {
        active = nextKey;
        refresh();
      });
      renderCards(cardRoot, items);
      countRoot.textContent = "Đang hiển thị " + items.length + " / " + data.length + " mẫu";
    }

    refresh();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPreview);
  } else {
    initPreview();
  }
})();
