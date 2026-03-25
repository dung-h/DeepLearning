(function () {
  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
      return;
    }
    fn();
  }

  ready(function () {
    var images = Array.prototype.slice.call(
      document.querySelectorAll(".figure-card img, .hard-case-card img")
    );

    if (!images.length) {
      return;
    }

    images.forEach(function (img) {
      img.classList.add("zoomable-media");
      img.setAttribute("tabindex", "0");
      img.setAttribute("title", "Nhấn để phóng to");
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", (img.getAttribute("alt") || "Hình ảnh") + " - nhấn để phóng to");
    });

    var lightbox = document.createElement("div");
    lightbox.className = "media-lightbox";
    lightbox.hidden = true;
    lightbox.innerHTML = [
      '<div class="media-lightbox-dialog" role="dialog" aria-modal="true" aria-label="Xem ảnh phóng to">',
      '<button type="button" class="media-lightbox-close" aria-label="Đóng ảnh phóng to">×</button>',
      '<figure class="media-lightbox-figure">',
      '<img class="media-lightbox-image" alt="">',
      '<figcaption class="media-lightbox-caption">',
      '<strong></strong>',
      '<span></span>',
      '</figcaption>',
      '</figure>',
      '</div>',
    ].join("");

    document.body.appendChild(lightbox);

    var dialog = lightbox.querySelector(".media-lightbox-dialog");
    var closeButton = lightbox.querySelector(".media-lightbox-close");
    var preview = lightbox.querySelector(".media-lightbox-image");
    var captionTitle = lightbox.querySelector(".media-lightbox-caption strong");
    var captionBody = lightbox.querySelector(".media-lightbox-caption span");
    var lastFocused = null;

    function getCaptionParts(img) {
      var card = img.closest(".figure-card, .hard-case-card");
      var title = "";
      var body = "";
      if (card) {
        var heading = card.querySelector("h3");
        var paragraphs = card.querySelectorAll("p");
        title = heading ? heading.textContent.trim() : (img.alt || "Hình ảnh");
        if (paragraphs.length) {
          body = paragraphs[0].textContent.trim();
        }
      }
      return {
        title: title || img.alt || "Hình ảnh",
        body: body || img.alt || "Ảnh trong báo cáo",
      };
    }

    function openLightbox(img) {
      var parts = getCaptionParts(img);
      lastFocused = document.activeElement;
      preview.src = img.currentSrc || img.src;
      preview.alt = img.alt || parts.title;
      captionTitle.textContent = parts.title;
      captionBody.textContent = parts.body;
      lightbox.hidden = false;
      document.body.style.overflow = "hidden";
      closeButton.focus();
    }

    function closeLightbox() {
      lightbox.hidden = true;
      preview.removeAttribute("src");
      document.body.style.overflow = "";
      if (lastFocused && typeof lastFocused.focus === "function") {
        lastFocused.focus();
      }
    }

    images.forEach(function (img) {
      img.addEventListener("click", function () {
        openLightbox(img);
      });
      img.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openLightbox(img);
        }
      });
    });

    closeButton.addEventListener("click", closeLightbox);
    lightbox.addEventListener("click", function (event) {
      if (event.target === lightbox) {
        closeLightbox();
      }
    });
    dialog.addEventListener("click", function (event) {
      event.stopPropagation();
    });
    document.addEventListener("keydown", function (event) {
      if (!lightbox.hidden && event.key === "Escape") {
        closeLightbox();
      }
    });
  });
})();
