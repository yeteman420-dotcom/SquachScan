/* ModToes Foot Scanner v3 — vanilla JS, no dependencies.
   Step flow: landing -> consent modal -> setup -> capture -> processing -> results.
   Consent ID is kept in memory only (never localStorage). */
(function () {
  "use strict";

  var state = {
    consentId: null,      // memory only
    consentTs: null,
    footSide: null,       // "left" | "right"
    reference: null,      // "quarter" | "card" | "none"
    stream: null,
    photoBlob: null,      // raw capture (from camera or file)
    result: null          // last successful /api/scan payload
  };

  // ---------- helpers ----------
  function $(id) { return document.getElementById(id); }

  function showStep(name) {
    var steps = document.querySelectorAll(".step");
    for (var i = 0; i < steps.length; i++) steps[i].classList.remove("active");
    var el = $("step-" + name);
    if (el) el.classList.add("active");
    window.scrollTo(0, 0);
    if (name !== "capture") stopCamera();
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach(function (t) { try { t.stop(); } catch (e) {} });
      state.stream = null;
    }
    var v = $("video");
    if (v) v.srcObject = null;
  }

  function fmt(n, digits) {
    return Number(n).toFixed(digits);
  }

  function scanDate() {
    var d = new Date();
    function p(x) { return (x < 10 ? "0" : "") + x; }
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate());
  }

  // ---------- consent ----------
  var backdrop = $("consent-backdrop");
  var checkbox = $("consent-checkbox");
  var btnConsent = $("btn-consent");
  var consentTsEl = $("consent-timestamp");

  $("btn-start").addEventListener("click", function () {
    checkbox.checked = false;
    btnConsent.disabled = true;
    consentTsEl.textContent = "Local time: " + new Date().toLocaleString();
    backdrop.classList.add("open");
  });

  checkbox.addEventListener("change", function () {
    btnConsent.disabled = !checkbox.checked;
  });

  $("btn-decline").addEventListener("click", function () {
    backdrop.classList.remove("open");
    showStep("landing");
  });

  btnConsent.addEventListener("click", function () {
    if (!checkbox.checked) return;
    btnConsent.disabled = true;
    btnConsent.textContent = "Recording…";
    fetch("/api/consent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ consent: true })
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data && data.ok && data.consent_id) {
          state.consentId = data.consent_id; // memory only — never persisted
          state.consentTs = new Date().toISOString();
          backdrop.classList.remove("open");
          showStep("setup");
        } else {
          throw new Error((data && data.error) || "Consent request failed");
        }
      })
      .catch(function (err) {
        alert("Couldn't record consent: " + err.message + "\nPlease try again.");
      })
      .then(function () {
        btnConsent.disabled = false;
        btnConsent.textContent = "I consent";
      });
  });

  // ---------- setup ----------
  function updateContinue() {
    $("btn-to-capture").disabled = !(state.footSide && state.reference);
    $("ref-none-note").hidden = state.reference !== "none";
  }

  var optBtns = document.querySelectorAll(".opt");
  for (var i = 0; i < optBtns.length; i++) {
    optBtns[i].addEventListener("click", function () {
      var group = this.getAttribute("data-group");
      var value = this.getAttribute("data-value");
      if (group === "foot") state.footSide = value;
      else if (group === "ref") state.reference = value;
      var siblings = document.querySelectorAll('.opt[data-group="' + group + '"]');
      for (var j = 0; j < siblings.length; j++) {
        siblings[j].classList.remove("selected");
        siblings[j].setAttribute("aria-pressed", "false");
      }
      this.classList.add("selected");
      this.setAttribute("aria-pressed", "true");
      updateContinue();
    });
  }

  $("btn-to-capture").addEventListener("click", function () {
    if (!state.consentId) { // safety: never scan without consent
      alert("Please accept consent first.");
      showStep("landing");
      return;
    }
    resetCaptureUI();
    showStep("capture");
    startCamera();
  });

  $("btn-back-setup").addEventListener("click", function () {
    showStep("setup");
  });

  // ---------- capture ----------
  function resetCaptureUI() {
    $("thumb-wrap").hidden = true;
    $("thumb").removeAttribute("src");
    state.photoBlob = null;
    var fc = $("file-camera"); if (fc) fc.value = "";
    var fu = $("file-upload"); if (fu) fu.value = "";
  }

  function guidesForReference() {
    var q = $("guide-quarter"), c = $("guide-card");
    if (state.reference === "card") {
      q.setAttribute("visibility", "hidden");
      c.setAttribute("visibility", "visible");
    } else {
      q.setAttribute("visibility", "visible");
      c.setAttribute("visibility", "hidden");
    }
  }

  function startCamera() {
    guidesForReference();
    var hasGUM = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    if (!hasGUM) { showFallback(); return; }
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    })
      .then(function (stream) {
        state.stream = stream;
        $("video").srcObject = stream;
        $("camera-ui").hidden = false;
        $("fallback-ui").hidden = true;
      })
      .catch(function () {
        // denied or unavailable -> fallback inputs
        showFallback();
      });
  }

  function showFallback() {
    $("camera-ui").hidden = true;
    $("fallback-ui").hidden = false;
  }

  $("btn-shutter").addEventListener("click", function () {
    var v = $("video");
    if (!v || !v.videoWidth) {
      alert("Camera isn't ready yet — please wait a moment and try again.");
      return;
    }
    var canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    canvas.getContext("2d").drawImage(v, 0, 0);
    canvas.toBlob(function (blob) {
      if (!blob) { alert("Couldn't capture the photo. Please try again."); return; }
      acceptPhoto(blob);
    }, "image/jpeg", 0.95);
  });

  $("btn-file-camera").addEventListener("click", function () { $("file-camera").click(); });
  $("btn-file-upload").addEventListener("click", function () { $("file-upload").click(); });

  function onFileChosen(e) {
    var f = e.target.files && e.target.files[0];
    if (!f) return;
    acceptPhoto(f);
  }
  $("file-camera").addEventListener("change", onFileChosen);
  $("file-upload").addEventListener("change", onFileChosen);

  function acceptPhoto(blob) {
    state.photoBlob = blob;
    var url = URL.createObjectURL(blob);
    $("thumb").src = url;
    $("thumb-wrap").hidden = false;
    $("thumb").onload = function () { URL.revokeObjectURL(url); };
  }

  $("btn-retake").addEventListener("click", function () {
    resetCaptureUI();
    startCamera();
  });

  // ---------- EXIF strip + downscale ----------
  // Drawing to a canvas and re-encoding drops ALL EXIF/GPS metadata.
  function stripExif(blob, cb) {
    var img = new Image();
    var url = URL.createObjectURL(blob);
    img.onload = function () {
      URL.revokeObjectURL(url);
      var w = img.naturalWidth, h = img.naturalHeight;
      var maxSide = Math.max(w, h);
      if (maxSide > 2000) {
        var s = 2000 / maxSide;
        w = Math.round(w * s); h = Math.round(h * s);
      }
      var canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      canvas.toBlob(cb, "image/jpeg", 0.92);
    };
    img.onerror = function () {
      URL.revokeObjectURL(url);
      cb(null);
    };
    img.src = url;
  }

  $("btn-process").addEventListener("click", function () {
    if (!state.photoBlob) { alert("No photo to process."); return; }
    if (!state.consentId) { alert("Consent is required before scanning."); return; }
    showStep("processing");
    stripExif(state.photoBlob, function (clean) {
      if (!clean) {
        showError("BAD_IMAGE", "We couldn't read that photo file. Please retake it or pick a different image.");
        return;
      }
      var fd = new FormData();
      fd.append("photo", clean, "foot.jpg");
      fd.append("foot_side", state.footSide);
      fd.append("reference", state.reference);
      fd.append("consent_id", state.consentId);
      fetch("/api/scan", { method: "POST", body: fd })
        .then(function (r) { return r.json(); })
        .then(handleScanResponse)
        .catch(function (err) {
          showError("NETWORK", "Couldn't reach the scanner: " + err.message + ". Check your connection and try again.");
        });
    });
  });

  // ---------- results ----------
  var ERROR_ADVICE = {
    NO_CONSENT: "Your consent record wasn't found. Please start a new scan and accept consent.",
    RATE_LIMITED: "Too many scans in a short time. Wait a few minutes and try again.",
    NO_FOOT_FOUND: "We couldn't find a foot in the photo. Retake it top-down with your whole bare foot clearly in frame on a plain contrasting floor.",
    NO_REFERENCE_FOUND: "We couldn't find your reference object. Retake with the quarter (or card) clearly visible beside your foot — not under it.",
    BAD_IMAGE: "That photo couldn't be read. Retake it with good, even light.",
    TOO_LARGE: "That photo is too large (max 8 MB). Try again — the app downsizes before upload.",
    NETWORK: null // custom message already provided
  };

  function handleScanResponse(data) {
    if (!data || data.ok !== true) {
      var code = (data && data.code) || "UNKNOWN";
      var msg = (data && data.error) || "The scan failed for an unknown reason.";
      showError(code, msg);
      return;
    }
    state.result = data;
    renderResults(data);
    showStep("results");
  }

  function showError(code, message) {
    var box = $("results-error");
    var advice = ERROR_ADVICE[code];
    if (advice === null) advice = message; // NETWORK carries its own text
    box.innerHTML = "";
    var h = document.createElement("h3");
    h.textContent = "Scan failed";
    var p = document.createElement("p");
    p.textContent = message;
    box.appendChild(h); box.appendChild(p);
    if (advice && advice !== message) {
      var a = document.createElement("p");
      a.innerHTML = "<strong>What to do:</strong> ";
      a.appendChild(document.createTextNode(advice));
      box.appendChild(a);
    }
    if (code !== "NETWORK") {
      var c = document.createElement("p");
      c.className = "small";
      c.innerHTML = "Error code: ";
      var cc = document.createElement("code");
      cc.textContent = code;
      c.appendChild(cc);
      box.appendChild(c);
    }
    box.hidden = false;
    $("results-body").hidden = true;
    $("results-warnings").hidden = true;
    showStep("results");
  }

  function renderResults(data) {
    $("results-error").hidden = true;
    var body = $("results-body");
    body.hidden = false;

    // --- warnings (amber, prominent) ---
    var warnBox = $("results-warnings");
    var warnList = $("warnings-list");
    warnList.innerHTML = "";
    var warnings = Array.isArray(data.warnings) ? data.warnings : [];
    if (warnings.length > 0) {
      for (var i = 0; i < warnings.length; i++) {
        var li = document.createElement("li");
        li.textContent = String(warnings[i]);
        warnList.appendChild(li);
      }
      warnBox.hidden = false;
    } else {
      warnBox.hidden = true;
    }

    // --- SVG via DOMParser ---
    var holder = $("svg-holder");
    holder.innerHTML = "";
    if (data.svg) {
      try {
        var parser = new DOMParser();
        var doc = parser.parseFromString(data.svg, "image/svg+xml");
        var svgEl = doc.documentElement;
        if (svgEl && svgEl.tagName.toLowerCase() === "svg") {
          holder.appendChild(document.importNode(svgEl, true));
        } else {
          holder.textContent = "(Drawing unavailable — SVG could not be parsed.)";
        }
      } catch (e) {
        holder.textContent = "(Drawing unavailable.)";
      }
    } else {
      holder.textContent = "(No drawing returned.)";
    }

    // --- PNG preview ---
    var png = $("png-preview");
    if (data.png_base64) {
      png.src = "data:image/png;base64," + data.png_base64;
      png.hidden = false;
    } else {
      png.hidden = true;
    }

    // --- measurements (null-safe) ---
    var m = data.measurements || {};
    var tbody = $("measure-body");
    tbody.innerHTML = "";
    var rows = [
      { key: "length_mm", label: "Foot length", inKey: "length_in" },
      { key: "ball_width_mm", label: "Ball width" },
      { key: "heel_width_mm", label: "Heel width" },
      { key: "arch_width_mm", label: "Arch width" }
    ];
    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      var tr = document.createElement("tr");
      var th = document.createElement("th");
      th.textContent = row.label;
      var td = document.createElement("td");
      td.className = "val";
      var mm = m[row.key];
      if (mm === null || mm === undefined || mm === "") {
        td.textContent = "—";
      } else {
        var txt = fmt(mm, 1) + " mm";
        if (row.inKey && m[row.inKey] !== null && m[row.inKey] !== undefined && m[row.inKey] !== "") {
          txt += " (" + fmt(m[row.inKey], 2) + " in)";
        }
        td.textContent = txt;
      }
      tr.appendChild(th); tr.appendChild(td);
      tbody.appendChild(tr);
    }

    // --- calibration note ---
    var note = $("calibration-note");
    if (m.calibrated === true) {
      var scaleTxt = (m.px_per_mm !== null && m.px_per_mm !== undefined)
        ? " Scale: " + fmt(m.px_per_mm, 2) + " px/mm"
        : "";
      var refTxt = m.reference ? " (" + m.reference + ")" : "";
      note.textContent = "Calibrated with reference object" + refTxt + "." + scaleTxt;
    } else {
      note.textContent = "Unscaled — no reference object. Shape only; measurements are not available. These drawings are estimates — always verify with physical measurement.";
    }

    // --- 3D model link (if foot length was measured) ---
    var row3d = $("row-3d");
    var btn3d = $("btn-view-3d");
    if (m.length_mm !== null && m.length_mm !== undefined && m.length_mm !== "" && m.length_mm > 0) {
      btn3d.href = "/3d?length_mm=" + encodeURIComponent(m.length_mm);
      row3d.hidden = false;
    } else {
      row3d.hidden = true;
    }

    // --- downloads ---
    var base = "modtoes-scan-" + scanDate() + "-" + (state.footSide || "foot");
    $("btn-dl-svg").onclick = function () {
      if (!state.result || !state.result.svg) return;
      var blob = new Blob([state.result.svg], { type: "image/svg+xml" });
      downloadBlob(blob, base + ".svg");
    };
    $("btn-dl-png").onclick = function () {
      if (!state.result || !state.result.png_base64) return;
      var bin = atob(state.result.png_base64);
      var bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      var blob = new Blob([bytes], { type: "image/png" });
      downloadBlob(blob, base + ".png");
    };
  }

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 500);
  }

  $("btn-again").addEventListener("click", function () {
    state.photoBlob = null;
    state.result = null;
    resetCaptureUI();
    showStep("setup"); // consent stays in memory — no need to re-consent
  });

  // Initial: landing is active in markup.
})();
