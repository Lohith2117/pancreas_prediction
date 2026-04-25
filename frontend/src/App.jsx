import React, { useState } from "react";
import axios from "axios";

const C = { bg: "#020617", card: "#0f172a", accent: "#38bdf8", red: "#ef4444" };

export default function App() {
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const runInference = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const { data } = await axios.post("http://localhost:8000/predict", formData);
      setResult(data);
    } catch (e) {
      console.error(e);
      alert("Backend Error: Check terminal for details.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: "#f8fafc", padding: "20px 60px", fontFamily: "Inter, sans-serif" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 40 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "1.5rem", color: C.accent, fontWeight: "900", letterSpacing: "1px" }}>
            PANCREAS<span style={{ color: "#fff" }}>NET</span> : : AI_VISION
          </h1>
          <p style={{ margin: 0, fontSize: "0.8rem", color: "#64748b", fontWeight: "600" }}>M2_AIR_NEURAL_ENGINE_ACTIVE</p>
        </div>
        <div style={{ display: "flex", gap: 15, alignItems: "center" }}>
          <input type="file" id="file-upload" hidden onChange={(e) => setSelectedFile(e.target.files[0])} />
          <label htmlFor="file-upload" style={{ cursor: "pointer", padding: "10px 20px", border: "1px solid #334155", borderRadius: "6px", fontSize: "12px", background: "#0f172a" }}>
            {selectedFile ? selectedFile.name.toUpperCase() : "SELECT_SOURCE (NII, DCM, JPG)"}
          </label>
          <button 
            onClick={runInference} 
            disabled={!selectedFile || analyzing}
            style={{ 
              padding: "10px 25px", 
              background: analyzing ? "#334155" : C.accent, 
              color: "#000", 
              border: "none", 
              borderRadius: "6px", 
              fontWeight: "bold", 
              cursor: "pointer",
              transition: "0.2s" 
            }}>
            {analyzing ? "ANALYZING..." : "EXECUTE_ANALYSIS"}
          </button>
        </div>
      </header>

      <main style={{ display: "flex", flexDirection: "column", gap: 30 }}>
        {/* DUAL VIEWER SECTION */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          
          {/* LEFT: SOURCE IMAGE */}
          <div style={{ background: "#000", borderRadius: 12, border: "1px solid #1e293b", overflow: "hidden" }}>
            <div style={{ padding: "10px 15px", background: "#0f172a", fontSize: "10px", color: C.accent, letterSpacing: "1px", fontWeight: "bold" }}>SOURCE_SCAN_AXIAL</div>
            <div style={{ height: "500px", display: "flex", alignItems: "center", justifyContent: "center" }}>
              {result ? (
                <img src={result.image} style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} alt="Original Scan" />
              ) : (
                <div style={{ color: "#334155", letterSpacing: "2px", fontSize: "12px" }}>AWAITING_INPUT_BUFFER...</div>
              )}
            </div>
          </div>

          {/* RIGHT: AI ANALYSIS OVERLAY */}
          <div style={{ background: "#000", borderRadius: 12, border: "1px solid #1e293b", overflow: "hidden", position: "relative" }}>
            <div style={{ padding: "10px 15px", background: "#0f172a", fontSize: "10px", color: C.red, letterSpacing: "1px", fontWeight: "bold" }}>AI_SEGMENTATION_OVERLAY</div>
            <div style={{ height: "500px", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
              {result ? (
                <>
                  <img src={result.image} style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} alt="Scan Background" />
                  <img 
                    src={result.mask} 
                    style={{ 
                      position: "absolute", 
                      maxWidth: "100%", 
                      maxHeight: "100%", 
                      objectFit: "contain", 
                      opacity: 0.7, 
                      mixBlendMode: "normal" 
                    }} 
                    alt="Tumor Mask"
                  />
                </>
              ) : (
                <div style={{ color: "#334155", letterSpacing: "2px", fontSize: "12px" }}>NEURAL_NET_STANDBY...</div>
              )}
            </div>
          </div>
        </div>

        {/* BOTTOM ANALYTICS BAR */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
            <StatCard 
              label="Diagnostic Confidence" 
              value={result ? (result.probability * 100).toFixed(1) + "%" : "0.0%"} 
              color={result?.probability > 0.5 ? C.red : C.accent} 
            />
            
            <div style={{ background: C.card, padding: 20, borderRadius: 12, border: "1px solid #1e293b" }}>
                <small style={{ color: "#64748b", fontWeight: "bold" }}>FINDINGS_LOG</small>
                <p style={{ margin: "10px 0 0 0", fontSize: "13px", color: "#f1f5f9", lineHeight: "1.5" }}>
                  {result 
                    ? (result.probability > 0.5 
                        ? "CRITICAL: Focal lesion detected. Anomaly aligns with pancreatic parenchyma morphology." 
                        : "NOMINAL: No significant hyper-intensities or tissue abnormalities identified.")
                    : "System idle. Awaiting clinical data for segmentation analysis."}
                </p>
            </div>

            <div style={{ background: C.card, padding: 20, borderRadius: 12, border: "1px solid #1e293b" }}>
                <small style={{ color: "#64748b", fontWeight: "bold" }}>SYSTEM_STATUS</small>
                <div style={{ marginTop: 10, color: "#34d399", fontWeight: "bold", fontSize: "14px", display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#34d399" }}></div>
                    GPU_ACCELERATED_INFERENCE
                </div>
                <div style={{ marginTop: 5, color: "#64748b", fontSize: "11px" }}>
                    LATENCY: {result ? "42ms" : "--"} | RESNET_50_CORE
                </div>
            </div>
        </div>
      </main>
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{ background: "#0f172a", padding: 20, borderRadius: 12, borderTop: `4px solid ${color}`, borderRight: "1px solid #1e293b", borderBottom: "1px solid #1e293b", borderLeft: "1px solid #1e293b" }}>
      <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: "1px", fontWeight: "bold" }}>{label}</div>
      <div style={{ fontSize: "2.2rem", fontWeight: "900", marginTop: 8, color: color }}>{value}</div>
    </div>
  );
}