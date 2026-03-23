(function () {
  var MONACO_BASE_URL = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs";
  var monacoLoaderPromise = null;

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function highlightPython(rawCode) {
    var tokens = [];
    var code = rawCode;

    function stash(pattern, className) {
      code = code.replace(pattern, function (match) {
        var id = "@@TOKEN_" + tokens.length + "@@";
        tokens.push('<span class="' + className + '">' + escapeHtml(match) + "</span>");
        return id;
      });
    }

    stash(/#[^\n]*/g, "token-comment");
    stash(/("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g, "token-string");

    code = escapeHtml(code);
    code = code.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="token-number">$1</span>');
    code = code.replace(
      /\b(def|class|return|if|else|elif|for|while|in|try|except|with|as|import|from|pass|continue|break|raise|yield|lambda|None|True|False|and|or|not|super)\b/g,
      '<span class="token-keyword">$1</span>'
    );
    code = code.replace(
      /\b(Path|DataLoader|Dataset|AutoTokenizer|AutoModelForSequenceClassification|CLIPModel|CLIPVisionModel|VisualBertModel|nn|torch|pd|np|Image)\b/g,
      '<span class="token-type">$1</span>'
    );

    code = code.replace(/@@TOKEN_(\d+)@@/g, function (_, index) {
      return tokens[Number(index)];
    });

    return code;
  }

  function createElement(tag, className, text) {
    var element = document.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (typeof text === "string") {
      element.textContent = text;
    }
    return element;
  }

  function ensureMonacoEnvironment() {
    if (window.MonacoEnvironment && window.MonacoEnvironment.__codexConfigured) {
      return;
    }

    window.MonacoEnvironment = {
      __codexConfigured: true,
      getWorkerUrl: function () {
        var workerScript =
          "self.MonacoEnvironment = { baseUrl: '" +
          MONACO_BASE_URL +
          "/' };" +
          "importScripts('" +
          MONACO_BASE_URL +
          "/base/worker/workerMain.js');";
        return "data:text/javascript;charset=utf-8," + encodeURIComponent(workerScript);
      },
    };
  }

  function loadMonaco() {
    if (window.monaco && window.monaco.editor) {
      return Promise.resolve(window.monaco);
    }

    if (monacoLoaderPromise) {
      return monacoLoaderPromise;
    }

    monacoLoaderPromise = new Promise(function (resolve, reject) {
      ensureMonacoEnvironment();

      function startLoader() {
        if (!window.require || !window.require.config) {
          reject(new Error("Monaco AMD loader is not available."));
          return;
        }

        window.require.config({
          paths: { vs: MONACO_BASE_URL },
          "vs/nls": { availableLanguages: { "*": "en" } },
        });

        window.require(
          ["vs/editor/editor.main"],
          function () {
            resolve(window.monaco);
          },
          function (error) {
            reject(error || new Error("Failed to load Monaco editor."));
          }
        );
      }

      if (window.require && window.require.config) {
        startLoader();
        return;
      }

      var script = document.createElement("script");
      script.src = MONACO_BASE_URL + "/loader.js";
      script.async = true;
      script.onload = startLoader;
      script.onerror = function () {
        reject(new Error("Failed to download Monaco loader."));
      };
      document.head.appendChild(script);
    });

    return monacoLoaderPromise;
  }

  function createFallbackFrame() {
    var codeWrap = createElement("div", "ide-code");
    var gutter = createElement("pre", "ide-gutter");
    var content = createElement("pre", "ide-content");
    codeWrap.append(gutter, content);
    return {
      node: codeWrap,
      gutter: gutter,
      content: content,
    };
  }

  function renderLines(container, code) {
    var lineCount = code.split("\n").length;
    var lineNodes = [];
    for (var index = 1; index <= lineCount; index += 1) {
      lineNodes.push('<span class="ide-line">' + index + "</span>");
    }
    container.innerHTML = lineNodes.join("");
  }

  function renderCode(container, code) {
    var highlighted = highlightPython(code);
    var html = highlighted
      .split("\n")
      .map(function (line) {
        return '<span class="ide-line">' + (line || "&nbsp;") + "</span>";
      })
      .join("");
    container.innerHTML = html;
  }

  function copyText(value) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(value);
    }

    return new Promise(function (resolve, reject) {
      var textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.setAttribute("readonly", "readonly");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand("copy");
        resolve();
      } catch (error) {
        reject(error);
      } finally {
        document.body.removeChild(textarea);
      }
    });
  }

  function getLanguage(file) {
    if (file.language) {
      return file.language;
    }

    var name = (file.name || "").toLowerCase();
    if (name.endsWith(".json")) {
      return "json";
    }
    if (name.endsWith(".md")) {
      return "markdown";
    }
    if (name.endsWith(".html")) {
      return "html";
    }
    if (name.endsWith(".css")) {
      return "css";
    }
    if (name.endsWith(".js")) {
      return "javascript";
    }
    if (name.endsWith(".ts")) {
      return "typescript";
    }
    return "python";
  }

  function defineMonacoTheme(monaco) {
    if (window.__REPORT_MONACO_THEME_DEFINED) {
      return;
    }

    monaco.editor.defineTheme("co3133-academic-dark", {
      base: "vs-dark",
      inherit: true,
      rules: [
        { token: "comment", foreground: "7f92b0" },
        { token: "string", foreground: "f9c784" },
        { token: "keyword", foreground: "8dc5ff" },
        { token: "number", foreground: "c7a7ff" },
        { token: "type.identifier", foreground: "8ce1b5" },
      ],
      colors: {
        "editor.background": "#0d1728",
        "editor.foreground": "#eef4ff",
        "editorLineNumber.foreground": "#6f83a0",
        "editorLineNumber.activeForeground": "#f3ead5",
        "editorIndentGuide.background1": "#243755",
        "editorIndentGuide.activeBackground1": "#3d5d8b",
        "editor.selectionBackground": "#1d4f8e55",
        "editor.inactiveSelectionBackground": "#1d4f8e2f",
        "editor.lineHighlightBackground": "#132036",
        "editorCursor.foreground": "#d8c08c",
        "editorWhitespace.foreground": "#243755",
      },
    });

    window.__REPORT_MONACO_THEME_DEFINED = true;
  }

  function buildViewer(root, config) {
    var activeIndex = 0;
    var monacoState = {
      ready: false,
      failed: false,
      editor: null,
      api: null,
      models: {},
      resizeObserver: null,
    };

    var topbar = createElement("div", "ide-topbar");
    var dots = createElement("div", "ide-dots");
    dots.innerHTML = "<span></span><span></span><span></span>";
    var title = createElement("div", "ide-title", config.title || "Trình xem mã nguồn");
    var toolbar = createElement("div", "ide-toolbar");
    var copyButton = createElement("button", "ide-copy", "Sao chép");
    copyButton.type = "button";
    toolbar.appendChild(copyButton);
    topbar.append(dots, title, toolbar);

    var tabs = createElement("div", "ide-tabs");
    var meta = createElement("div", "ide-meta");
    var metaCopy = createElement("div", "ide-meta-copy");
    var metaBadges = createElement("div", "ide-meta-badges");
    meta.append(metaCopy, metaBadges);

    var stage = createElement("div", "ide-stage");
    var loading = createElement("div", "ide-loading", "Đang tải Monaco Editor...");
    stage.appendChild(loading);

    var footer = createElement("div", "ide-footer");
    var footerLeft = createElement("span", "", config.footer || "");
    var footerRight = createElement("span", "", "");
    footer.append(footerLeft, footerRight);

    root.innerHTML = "";
    root.classList.add("ide-viewer");
    root.append(topbar, tabs, meta, stage, footer);

    var fallbackFrame = createFallbackFrame();

    function ensureFallback(message) {
      if (!monacoState.failed) {
        monacoState.failed = true;
      }

      stage.innerHTML = "";

      if (message) {
        var note = createElement("div", "ide-loading ide-loading-compact", message);
        stage.appendChild(note);
      }

      stage.appendChild(fallbackFrame.node);
    }

    function getMonacoModel(file) {
      if (monacoState.models[file.name]) {
        return monacoState.models[file.name];
      }

      var uri = monacoState.api.Uri.parse(
        "inmemory://co3133/" + encodeURIComponent(file.name)
      );
      var model = monacoState.api.editor.createModel(
        file.code,
        getLanguage(file),
        uri
      );
      monacoState.models[file.name] = model;
      return model;
    }

    function ensureMonacoStage(monaco) {
      if (monacoState.editor) {
        return;
      }

      monacoState.api = monaco;
      defineMonacoTheme(monaco);
      stage.innerHTML = "";

      var shell = createElement("div", "ide-monaco-shell");
      var host = createElement("div", "ide-monaco");
      shell.appendChild(host);
      stage.appendChild(shell);

      monacoState.editor = monaco.editor.create(host, {
        value: "",
        language: "python",
        readOnly: true,
        theme: "co3133-academic-dark",
        automaticLayout: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        fontFamily: "IBM Plex Mono, Courier New, monospace",
        fontSize: 14,
        fontLigatures: false,
        lineHeight: 25,
        lineNumbersMinChars: 3,
        glyphMargin: false,
        folding: true,
        smoothScrolling: true,
        renderLineHighlight: "all",
        roundedSelection: false,
        wordWrap: "off",
        padding: { top: 16, bottom: 16 },
        overviewRulerLanes: 0,
        scrollbar: {
          verticalScrollbarSize: 10,
          horizontalScrollbarSize: 10,
        },
      });

      if (window.ResizeObserver) {
        monacoState.resizeObserver = new ResizeObserver(function () {
          if (monacoState.editor) {
            monacoState.editor.layout();
          }
        });
        monacoState.resizeObserver.observe(host);
      }
    }

    function update(index) {
      activeIndex = index;
      var current = config.files[index];

      Array.prototype.forEach.call(
        tabs.querySelectorAll(".ide-tab"),
        function (button, buttonIndex) {
          button.classList.toggle("is-active", buttonIndex === index);
        }
      );

      metaCopy.textContent = current.summary;
      metaBadges.innerHTML = "";
      (current.badges || []).forEach(function (badge) {
        var badgeNode = createElement("span", "", badge);
        metaBadges.appendChild(badgeNode);
      });

      if (monacoState.ready && monacoState.editor && monacoState.api) {
        var model = getMonacoModel(current);
        monacoState.editor.setModel(model);
        monacoState.editor.setScrollTop(0);
        monacoState.editor.setScrollLeft(0);
        footerRight.textContent =
          (current.source || current.name) + " | " + model.getLineCount() + " dòng";
        return;
      }

      if (monacoState.failed) {
        renderLines(fallbackFrame.gutter, current.code);
        renderCode(fallbackFrame.content, current.code);
        footerRight.textContent =
          (current.source || current.name) + " | " + current.code.split("\n").length + " dòng";
        return;
      }

      loading.textContent = "Đang tải trình xem cho " + (current.source || current.name) + "...";
      footerRight.textContent =
        (current.source || current.name) + " | " + current.code.split("\n").length + " dòng";
    }

    config.files.forEach(function (file, index) {
      var button = createElement("button", "ide-tab", file.tabLabel || file.name);
      button.type = "button";
      button.addEventListener("click", function () {
        update(index);
      });
      tabs.appendChild(button);
    });

    copyButton.addEventListener("click", function () {
      var current = config.files[activeIndex];
      copyText(current.code).then(function () {
        copyButton.textContent = "Đã sao chép";
        window.setTimeout(function () {
          copyButton.textContent = "Sao chép";
        }, 1200);
      });
    });

    update(0);

    loadMonaco()
      .then(function (monaco) {
        monacoState.ready = true;
        ensureMonacoStage(monaco);
        update(activeIndex);
      })
      .catch(function () {
        ensureFallback("Không tải được Monaco. Đang dùng trình đọc dự phòng.");
        update(activeIndex);
      });
  }

  document.querySelectorAll("[data-code-viewer]").forEach(function (node) {
    var viewerId = node.getAttribute("data-code-viewer");
    if (!window.REPORT_CODE_DATA || !window.REPORT_CODE_DATA[viewerId]) {
      return;
    }
    buildViewer(node, window.REPORT_CODE_DATA[viewerId]);
  });
})();

