import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API = "http://127.0.0.1:8000";
const MAX_QUESTION_LENGTH = 2000;

const css = `
  .title { margin: 0 0 4px; font-size: 30px; }
  .subtitle { color: #9aa3b5; margin: 0 0 24px; }
  .muted { color: #9aa3b5; font-size: 14px; margin: 8px 0; }
  .status { font-size: 14px; margin: 8px 0; }
  .status.ok { color: #6fdc8c; }
  .status.err { color: #ff7b7b; }
  .status.info { color: #9aa3b5; }
  .card { background: #181b24; border: 1px solid #2a2f3d; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
  .card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .link { background: none; border: none; color: #7f9bff; cursor: pointer; font-size: 14px; margin-left: 12px; }
  .link:hover { text-decoration: underline; }
  .doc-list { max-height: 200px; overflow-y: auto; border: 1px solid #2a2f3d; border-radius: 8px; padding: 6px 12px; background: #0f1117; }
  .doc { display: flex; align-items: center; gap: 10px; padding: 6px 0; cursor: pointer; font-size: 15px; }
  .upload { margin-top: 14px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 8px; }
  .tab { padding: 8px 18px; border-radius: 8px; border: 1px solid #3a4152; background: #252a38; color: #e6e6e6; cursor: pointer; font-size: 15px; }
  .tab.active { background: #4f6df5; border-color: #4f6df5; color: #fff; font-weight: 600; }
  .row { display: flex; gap: 10px; }
  .input { flex: 1; padding: 12px; font-size: 16px; border-radius: 8px; border: 1px solid #3a4152; background: #0f1117; color: #e6e6e6; }
  .btn { padding: 12px 22px; border-radius: 8px; border: none; background: #4f6df5; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #6280ff; }
  .btn:disabled { opacity: 0.6; cursor: not-allowed; }
  .answer { overflow-x: auto; line-height: 1.6; }
  .answer.error { color: #ff7b7b; }
  .answer table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
  .answer th, .answer td { border: 1px solid #3a4152; padding: 8px; text-align: left; vertical-align: top; }
  .answer th { background: #252a38; }
  .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 14px; color: #9aa3b5; }
  .chip { background: #252a38; border: 1px solid #3a4152; border-radius: 999px; padding: 3px 12px; color: #e6e6e6; }
`;

// Turns a FastAPI error response into a readable message.
// "detail" is a string for our own errors and a list for validation errors.
async function errorMessage(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      return data.detail.map((d) => d.msg).join("; ");
    }
  } catch {
    // response had no JSON body
  }
  return `Request failed (error ${response.status}).`;
}

function App() {
  const [mode, setMode] = useState("ask");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [answerMode, setAnswerMode] = useState("ask");
  const [isError, setIsError] = useState(false);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ text: "", kind: "info" });
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]);

  const loadDocuments = async () => {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error(await errorMessage(res));
      const data = await res.json();
      const docs = data.documents || [];
      setDocuments(docs);
      setSelected(docs);
    } catch (error) {
      setDocuments([]);
      setSelected([]);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const toggleDoc = (name) => {
    setSelected((prev) =>
      prev.includes(name) ? prev.filter((d) => d !== name) : [...prev, name]
    );
  };

  const showError = (message) => {
    setIsError(true);
    setAnswer(message);
    setSources([]);
  };

  const askQuestion = async () => {
    if (loading || !question.trim()) return;

    if (mode === "compare" && selected.length < 2) {
      setAnswerMode(mode);
      showError("Please tick at least 2 documents above to compare.");
      return;
    }

    setLoading(true);
    setAnswer("");
    setSources([]);
    setIsError(false);
    setAnswerMode(mode);

    try {
      const endpoint = mode === "ask" ? "/ask" : "/compare";
      // Ask: the backend chooses how many chunks to retrieve (top_k default).
      const body = mode === "ask" ? { question } : { question, documents: selected };

      const response = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        showError(await errorMessage(response));
      } else {
        const data = await response.json();
        setAnswer(data.answer || "No answer was returned.");
        setSources((mode === "ask" ? data.sources : data.documents_compared) || []);
      }
    } catch (error) {
      showError("Could not reach the backend. Is the server running?");
    }

    setLoading(false);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadStatus({ text: `Uploading ${file.name}...`, kind: "info" });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        setUploadStatus({ text: `❌ ${file.name}: ${await errorMessage(response)}`, kind: "err" });
      } else {
        const data = await response.json();
        setUploadStatus({
          text: `✅ ${data.filename} processed (${data.chunks_stored} chunks stored)`,
          kind: "ok",
        });
        await loadDocuments();
      }
    } catch (error) {
      setUploadStatus({ text: "❌ Upload failed. Is the backend running?", kind: "err" });
    }

    e.target.value = "";
  };

  return (
    <div style={{ maxWidth: "860px", margin: "0 auto", padding: "40px 20px" }}>
      <style>{css}</style>

      <h1 className="title">AI Study Assistant</h1>
      <p className="subtitle">
        Upload your PDFs, ask questions, or compare documents side by side.
      </p>

      <div className="card">
        <div className="card-head">
          <strong>
            Documents ({selected.length}/{documents.length} selected)
          </strong>
          <span>
            <button className="link" onClick={() => setSelected(documents)}>Select all</button>
            <button className="link" onClick={() => setSelected([])}>Clear</button>
          </span>
        </div>

        <div className="doc-list">
          {documents.length === 0 && (
            <p className="muted">No documents yet. Upload a PDF below.</p>
          )}
          {documents.map((d) => (
            <label key={d} className="doc">
              <input
                type="checkbox"
                checked={selected.includes(d)}
                onChange={() => toggleDoc(d)}
              />
              {d}
            </label>
          ))}
        </div>

        <div className="upload">
          <input type="file" accept=".pdf" onChange={handleFileUpload} />
          {uploadStatus.text && (
            <p className={`status ${uploadStatus.kind}`}>{uploadStatus.text}</p>
          )}
        </div>
      </div>

      <div className="card">
        <div className="tabs">
          <button className={`tab ${mode === "ask" ? "active" : ""}`} onClick={() => setMode("ask")}>
            Ask
          </button>
          <button className={`tab ${mode === "compare" ? "active" : ""}`} onClick={() => setMode("compare")}>
            Compare
          </button>
        </div>
        <p className="muted">
          {mode === "ask"
            ? "Ask a question across all uploaded documents."
            : "Compare only the documents ticked above."}
        </p>
        <div className="row">
          <input
            className="input"
            type="text"
            value={question}
            maxLength={MAX_QUESTION_LENGTH}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && askQuestion()}
            placeholder={mode === "ask" ? "Ask a question..." : "What do you want to compare?"}
          />
          <button className="btn" onClick={askQuestion} disabled={loading}>
            {loading ? "Thinking..." : mode === "ask" ? "Ask" : "Compare"}
          </button>
        </div>
      </div>

      {answer && (
        <div className="card">
          <div className={`answer ${isError ? "error" : ""}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
          </div>
          {sources.length > 0 && (
            <div className="chips">
              <strong>{answerMode === "ask" ? "Sources:" : "Compared:"}</strong>
              {sources.map((s) => (
                <span key={s} className="chip">{s}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;