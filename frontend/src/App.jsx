import { useState, useEffect, useMemo, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API = "http://127.0.0.1:8000";
const MAX_QUESTION_LENGTH = 2000;
const REPO_URL = "https://github.com/Prerit-kumar/ai-study-assistant";

// Shelf sections for the demo library. Uploaded papers that aren't listed go to "Your uploads".
const TOPICS = [
  { id: "attention", name: "Attention and long context", color: "var(--c-attention)",
    papers: ["Attention_Is_All_You_Need", "Transformer_XL", "Longformer", "BigBird", "Linformer", "Performer", "Reformer", "FlashAttention", "Mamba"] },
  { id: "encoders", name: "Pretrained encoders", color: "var(--c-encoders)",
    papers: ["BERT", "RoBERTa", "ALBERT", "DistilBERT", "ELECTRA", "Sentence_BERT", "T5"] },
  { id: "llms", name: "LLMs and scaling", color: "var(--c-llms)",
    papers: ["GPT3_Few_Shot_Learners", "Scaling_Laws", "Chinchilla", "LLaMA", "Llama2", "Mistral_7B", "Mixtral", "Switch_Transformers"] },
  { id: "retrieval", name: "Retrieval", color: "var(--c-retrieval)",
    papers: ["DPR_Dense_Passage_Retrieval", "REALM", "RAG_Retrieval_Augmented_Generation", "ColBERT", "RETRO", "Self_RAG", "Lost_in_the_Middle"] },
  { id: "tuning", name: "Fine-tuning and alignment", color: "var(--c-tuning)",
    papers: ["Adapters_Parameter_Efficient", "Prefix_Tuning", "LoRA", "QLoRA", "InstructGPT", "DPO"] },
  { id: "reasoning", name: "Reasoning and tools", color: "var(--c-reasoning)",
    papers: ["Chain_of_Thought", "ReAct", "Toolformer"] },
];
const UPLOADS = { id: "uploads", name: "Your uploads", color: "var(--c-uploads)" };

const EXAMPLES = {
  ask: [
    "How many parameters does GPT-3 have?",
    "What does LoRA freeze during fine-tuning?",
    "Why is FlashAttention faster than standard attention?",
  ],
  compare: [
    "Compare how these papers reduce the complexity of self-attention. Give a table.",
    "Compare the parameter-efficient fine-tuning methods in these papers.",
    "Compare how these papers use retrieval to improve language models.",
  ],
};

const css = `
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
  --paper: #eef0f3;
  --surface: #ffffff;
  --ink: #1b2230;
  --muted: #5b6474;
  --rule: #d6dbe3;
  --accent: #24527a;
  --accent-ink: #ffffff;
  --accent-deep: #163a59;
  --extrude: #c3cad6;
  --light: rgba(255,255,255,0.75);
  --highlight: #fce38a;
  --highlight-ink: #1b2230;
  --wood: #9b7752;
  --wood-edge: #6f5236;
  --wood-top: #b8946b;
  --pages: #f4efe2;
  --pages-line: #d9d0bb;
  --depth: 62px;
  --danger: #b3261e;
  --ok: #2e6b3f;
  --c-attention: #2f4a6d;
  --c-encoders: #7a2e35;
  --c-llms: #2f5d46;
  --c-retrieval: #a0712a;
  --c-tuning: #5b3f6b;
  --c-reasoning: #2c6670;
  --c-uploads: #6b7280;
  --serif: "Source Serif 4", Georgia, serif;
  --sans: "Instrument Sans", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #14181e;
    --surface: #1c222b;
    --ink: #e6e9ee;
    --muted: #9aa3b2;
    --rule: #2d3440;
    --accent: #8db4e2;
    --accent-ink: #10161e;
    --accent-deep: #4f7cae;
    --extrude: #0b0e13;
    --light: rgba(255,255,255,0.05);
    --highlight: #e8cc5c;
    --wood: #6d5236;
    --wood-edge: #4a3622;
    --wood-top: #806244;
    --pages: #cfc8b6;
    --pages-line: #a9a18e;
    --danger: #f2a49e;
    --ok: #93d3a2;
    --c-attention: #3d5f8a;
    --c-encoders: #94414a;
    --c-llms: #3c7559;
    --c-retrieval: #b5843a;
    --c-tuning: #74548a;
    --c-reasoning: #3a8290;
  }
}

* { box-sizing: border-box; }
body { margin: 0; background: radial-gradient(1100px 520px at 50% -12%, var(--light), transparent 70%) no-repeat, var(--paper); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; }
button, input { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.page { max-width: 1200px; margin: 0 auto; padding: 28px 24px 72px; }
@media (max-width: 640px) { .page { padding: 18px 16px 56px; } }

/* ---------- header ---------- */
.masthead { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.masthead h1 {
  font-family: var(--serif); font-weight: 600; font-size: 44px; line-height: 1.05; margin: 0; letter-spacing: -0.015em;
  text-shadow: 1px 1px 0 var(--extrude), 2px 2px 0 var(--extrude), 3px 3px 0 var(--extrude), 4px 4px 0 var(--extrude),
    5px 5px 0 var(--extrude), 8px 10px 14px rgba(0,0,0,0.18);
}
@media (max-width: 640px) { .masthead h1 { font-size: 34px; } }
.masthead p { margin: 10px 0 0; color: var(--muted); max-width: 62ch; font-size: 16px; }
.masthead a { color: var(--accent); font-weight: 500; text-decoration: none; }
.masthead a:hover { text-decoration: underline; }

/* ---------- bookshelf ---------- */
.shelf-wrap { margin-top: 26px; background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 16px 18px 14px; box-shadow: 0 1px 0 var(--rule), 0 4px 0 -2px var(--surface), 0 5px 0 -2px var(--rule), 0 8px 0 -4px var(--surface), 0 9px 0 -4px var(--rule), 0 24px 34px -22px rgba(0,0,0,0.45); }
.shelf-bar { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; justify-content: space-between; }
.shelf-bar .hint { color: var(--muted); font-size: 14px; margin: 0; }
.shelf-tools { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.filter { width: 200px; padding: 7px 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--paper); }
.text-btn { background: none; border: 0; padding: 0; color: var(--accent); cursor: pointer; font-size: 14px; font-weight: 500; }
.text-btn:hover { text-decoration: underline; }

.shelf { perspective: 1300px; perspective-origin: 50% -30%; padding: 30px 28px 34px; }
.shelf.dragging { outline: 2px dashed var(--accent); outline-offset: 4px; border-radius: 8px; }
.bookcase {
  display: flex; flex-wrap: wrap; gap: 46px 34px; transform-style: preserve-3d;
  transform: rotateX(var(--rx, 10deg)) rotateY(var(--ry, 0deg));
  transition: transform 450ms cubic-bezier(.2,.7,.2,1);
}
@media (max-width: 640px) { .shelf { overflow-x: auto; } .bookcase { flex-wrap: nowrap; } }
.section { flex: 0 0 auto; display: flex; flex-direction: column; transform-style: preserve-3d; }
.spines { position: relative; display: flex; align-items: flex-end; gap: 3px; height: 160px; padding: 0 10px; transform-style: preserve-3d; }
/* shelf plank: top surface (lies flat, going back) and front edge */
.spines::before {
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: var(--depth);
  transform-origin: bottom; transform: rotateX(90deg);
  background: linear-gradient(90deg, rgba(0,0,0,0.06) 0 1px, transparent 1px 40px) 0 0 / 41px 100%, var(--wood-top);
}
.spines::after {
  content: ""; position: absolute; left: 0; right: 0; top: 100%; height: 13px;
  background: linear-gradient(var(--wood), var(--wood-edge)); border-radius: 0 0 3px 3px;
  box-shadow: 0 10px 16px -6px rgba(0,0,0,0.35);
}
.section-name { font-size: 12px; color: var(--muted); margin-top: 22px; white-space: nowrap; }

.spine {
  --book: var(--spine);
  position: relative; width: 27px; height: var(--h); border: 0; padding: 0; cursor: pointer; color: #fff;
  border-radius: 2px 2px 0 0; transform-style: preserve-3d; transform-origin: bottom center;
  background:
    linear-gradient(rgba(255,255,255,0.45), rgba(255,255,255,0.45)) no-repeat 50% 10px / calc(100% - 8px) 2px,
    linear-gradient(rgba(255,255,255,0.3), rgba(255,255,255,0.3)) no-repeat 50% calc(100% - 12px) / calc(100% - 8px) 2px,
    linear-gradient(90deg, rgba(255,255,255,0.16), transparent 35%, rgba(0,0,0,0.18)),
    var(--book);
  transition: transform 260ms cubic-bezier(.2,.7,.2,1), box-shadow 260ms ease, color 200ms ease;
}
/* top of the book: page edges, going back into the shelf */
.spine::before {
  content: ""; position: absolute; left: 0; top: 0; width: 100%; height: var(--depth);
  transform-origin: top; transform: rotateX(-90deg);
  background: repeating-linear-gradient(90deg, var(--pages) 0 2px, var(--pages-line) 2px 3px);
  box-shadow: inset 0 0 0 2px var(--book);
}
/* right side of the book: the cover, visible when a book is pulled out */
.spine::after {
  content: ""; position: absolute; top: 0; left: 100%; width: var(--depth); height: 100%;
  transform-origin: left; transform: rotateY(90deg);
  background: linear-gradient(90deg, rgba(0,0,0,0.38), rgba(0,0,0,0.15)), var(--book);
}
.spine span {
  position: absolute; inset: 18px 0 18px 0; writing-mode: vertical-rl; transform: rotate(180deg);
  font-size: 11.5px; font-weight: 500; letter-spacing: 0.02em; line-height: 27px; text-align: left;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.spine:hover, .spine:focus-visible { transform: translateZ(38px) translateY(-6px) rotateY(-16deg); z-index: 2; }
.compare-mode .spine.off { --book: color-mix(in srgb, var(--spine) 28%, var(--surface)); color: var(--muted); transform: translateZ(-14px); }
.compare-mode .spine.on { transform: translateZ(10px); }
.compare-mode .spine.on:hover, .compare-mode .spine.off:hover { transform: translateZ(38px) translateY(-6px) rotateY(-16deg); }
.page .spine.dim { --book: color-mix(in srgb, var(--spine) 18%, var(--surface)); color: transparent; }
.page .spine.cited { transform: translateZ(46px) translateY(-8px) rotateY(-10deg); box-shadow: 0 0 0 3px var(--highlight), 0 16px 26px rgba(0,0,0,0.3); z-index: 3; }

.add-spine { width: 27px; height: 128px; border: 2px dashed var(--rule); border-radius: 2px 2px 0 0; background: none; color: var(--muted); font-size: 20px; cursor: pointer; align-self: flex-end; }
.add-spine:hover { border-color: var(--accent); color: var(--accent); }

.caption { min-height: 22px; margin: 14px 0 0; font-size: 14px; color: var(--muted); }
.caption strong { color: var(--ink); font-weight: 600; }
.status { font-size: 14px; margin: 4px 0 0; }
.status.ok { color: var(--ok); }
.status.err { color: var(--danger); }

@media (prefers-reduced-motion: no-preference) {
  .shelf.intro .spine { animation: drop 620ms cubic-bezier(.3,1.4,.5,1) backwards; animation-delay: calc(var(--i) * 28ms); }
  @keyframes drop { from { transform: translateY(-90px) translateZ(30px) rotate(-7deg); } }
  .masthead h1 { animation: rise 700ms cubic-bezier(.2,.7,.2,1) backwards; }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); text-shadow: none; } }
  .shelf.scanning .spine { animation: scan 1.3s ease-in-out infinite; animation-delay: calc(var(--i) * 45ms); }
  @keyframes scan { 0%, 60%, 100% { transform: translateZ(0); } 30% { transform: translateZ(30px) translateY(-6px); } }
}
@media (prefers-reduced-motion: reduce) { .bookcase, .spine { transition: none; } }

/* ---------- desk ---------- */
.desk { margin-top: 28px; max-width: 920px; }
.modes { display: inline-flex; gap: 8px; }
.modes button {
  border: 1px solid var(--rule); background: var(--surface); padding: 8px 22px; cursor: pointer; font-weight: 600; color: var(--muted);
  border-radius: 8px; box-shadow: 0 4px 0 var(--rule); transform: translateY(0); transition: transform 120ms ease, box-shadow 120ms ease;
}
.modes button:hover { transform: translateY(-1px); box-shadow: 0 5px 0 var(--rule); }
.modes button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent-deep); box-shadow: 0 1px 0 var(--accent-deep); transform: translateY(3px); }
.mode-note { color: var(--muted); margin: 10px 0 14px; }

.ask-row { display: flex; gap: 10px; }
.question { flex: 1; min-width: 0; padding: 15px 16px; font-size: 16px; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); box-shadow: inset 0 2px 4px rgba(0,0,0,0.08); }
.primary {
  padding: 0 26px; border: 0; border-radius: 8px; background: var(--accent); color: var(--accent-ink); font-weight: 600; cursor: pointer; white-space: nowrap;
  box-shadow: 0 5px 0 var(--accent-deep), 0 12px 18px -10px rgba(0,0,0,0.45); transition: transform 120ms ease, box-shadow 120ms ease;
}
.primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 0 var(--accent-deep), 0 14px 20px -10px rgba(0,0,0,0.45); }
.primary:active:not(:disabled) { transform: translateY(4px); box-shadow: 0 1px 0 var(--accent-deep); }
.primary:disabled { opacity: 0.55; cursor: progress; }
@media (max-width: 520px) { .ask-row { flex-direction: column; } .primary { padding: 12px; } }

.examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.examples span { color: var(--muted); font-size: 13px; align-self: center; }
.example {
  border: 1px solid var(--rule); background: var(--surface); border-radius: 4px; padding: 7px 12px; font-size: 13px; cursor: pointer; text-align: left;
  background-image: linear-gradient(var(--accent), var(--accent)); background-size: 100% 2px; background-repeat: no-repeat; background-position: 0 0;
  box-shadow: 0 2px 0 var(--rule), 0 8px 14px -10px rgba(0,0,0,0.4);
  transform: rotate(var(--tilt, -0.8deg)); transition: transform 160ms ease, box-shadow 160ms ease, color 160ms ease;
}
.example:nth-of-type(2n) { --tilt: 0.7deg; }
.example:nth-of-type(3n) { --tilt: -0.4deg; }
.example:hover { transform: translateY(-4px) rotate(0deg); color: var(--accent); box-shadow: 0 2px 0 var(--rule), 0 16px 20px -12px rgba(0,0,0,0.45); }

/* ---------- result ---------- */
.result { margin-top: 30px; background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 22px 24px; transform-origin: top center; box-shadow: 0 1px 0 var(--rule), 0 4px 0 -2px var(--surface), 0 5px 0 -2px var(--rule), 0 8px 0 -4px var(--surface), 0 9px 0 -4px var(--rule), 0 24px 34px -22px rgba(0,0,0,0.45); }
@media (prefers-reduced-motion: no-preference) {
  .result.fresh { animation: unfold 520ms cubic-bezier(.2,.7,.2,1); }
  @keyframes unfold { from { opacity: 0; transform: perspective(1000px) rotateX(-9deg) translateY(10px); } to { opacity: 1; transform: none; } }
}
.result-grid { display: grid; grid-template-columns: minmax(0, 1fr) 190px; gap: 28px; }
.result-grid.wide { grid-template-columns: 1fr; }
@media (max-width: 900px) { .result-grid { grid-template-columns: 1fr; } }

.answer { font-family: var(--serif); font-size: 17px; line-height: 1.65; overflow-x: auto; }
.answer p { margin: 0 0 0.9em; max-width: 72ch; }
.answer table { border-collapse: collapse; width: 100%; margin: 4px 0 18px; font-family: var(--sans); font-size: 14px; line-height: 1.45; }
.answer th, .answer td { border-bottom: 1px solid var(--rule); padding: 9px 10px; text-align: left; vertical-align: top; }
.answer th { font-weight: 600; border-bottom: 2px solid var(--ink); }
.answer.error { font-family: var(--sans); font-size: 15px; color: var(--danger); }
.loading { font-family: var(--serif); font-size: 17px; color: var(--muted); margin: 0; }

.margin h3 { font-size: 13px; font-weight: 600; color: var(--muted); margin: 0 0 8px; }
.margin ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 7px; }
.result-grid.wide .margin ul { flex-direction: row; flex-wrap: wrap; gap: 6px 12px; }
.margin li { font-size: 14px; overflow-wrap: anywhere; }
.margin mark {
  color: var(--highlight-ink); padding: 1px 4px; border-radius: 2px;
  background: linear-gradient(var(--highlight), var(--highlight)) no-repeat left / 100% 100%;
  box-decoration-break: clone; -webkit-box-decoration-break: clone;
}
@media (prefers-reduced-motion: no-preference) {
  .result.fresh .margin mark { animation: swipe 520ms ease-out both; animation-delay: calc(300ms + var(--i) * 120ms); }
  @keyframes swipe { from { background-size: 0% 100%; } to { background-size: 100% 100%; } }
}

.notes { margin-top: 18px; border-top: 1px solid var(--rule); padding-top: 12px; font-size: 14px; color: var(--muted); }
.notes summary { cursor: pointer; color: var(--ink); font-weight: 500; }
.notes p { margin: 8px 0 0; }
.result-foot { display: flex; justify-content: flex-end; margin-top: 14px; }
`;

const prettyName = (doc) => doc.replace(/\.pdf$/i, "").replace(/_/g, " ");

// Gives every book a stable, slightly different height.
function spineHeight(name) {
  let hash = 0;
  for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) % 997;
  return 120 + (hash % 30);
}

async function errorMessage(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail) && data.detail.length > 0) return data.detail.map((d) => d.msg).join("; ");
  } catch {
    // no JSON body
  }
  return `Request failed (error ${response.status}).`;
}

// Compare answers end with "---" followed by notes about left-out papers.
function splitNotes(text) {
  const index = text.lastIndexOf("\n---\n");
  if (index === -1) return [text, ""];
  return [text.slice(0, index), text.slice(index + 5).replace(/^\s+/, "")];
}

function App() {
  const [mode, setMode] = useState("ask");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null); // { mode, text, sources, isError }
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ text: "", kind: "" });
  const [dragging, setDragging] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]);
  const [filter, setFilter] = useState("");
  const [hovered, setHovered] = useState(null);
  const [tilt, setTilt] = useState(null);
  const [intro, setIntro] = useState(true);
  const fileInput = useRef(null);
  const questionInput = useRef(null);

  const loadDocuments = async () => {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error(await errorMessage(res));
      const data = await res.json();
      const docs = data.documents || [];
      setDocuments(docs);
      setSelected(docs);
    } catch {
      setDocuments([]);
      setSelected([]);
    }
  };

  useEffect(() => {
    loadDocuments();
    const timer = setTimeout(() => setIntro(false), 2600);
    return () => clearTimeout(timer);
  }, []);

  // Group the documents into shelf sections.
  const sections = useMemo(() => {
    const placed = new Set();
    const result = TOPICS.map((topic) => {
      const docs = topic.papers.map((p) => `${p}.pdf`).filter((d) => documents.includes(d));
      docs.forEach((d) => placed.add(d));
      return { ...topic, docs };
    }).filter((t) => t.docs.length > 0);
    const rest = documents.filter((d) => !placed.has(d));
    if (rest.length > 0) result.push({ ...UPLOADS, docs: rest });
    return result;
  }, [documents]);

  const term = filter.trim().toLowerCase();
  const cited = new Set(result && !result.isError ? result.sources : []);

  const run = async (text = question, runMode = mode) => {
    const q = text.trim();
    if (loading || !q) return;

    if (runMode === "compare" && selected.length < 2) {
      setResult({ mode: runMode, text: "Pick at least 2 books on the shelf to compare them.", sources: [], isError: true });
      return;
    }

    setLoading(true);
    setCopied(false);
    setResult(null);

    try {
      const endpoint = runMode === "ask" ? "/ask" : "/compare";
      const body = runMode === "ask" ? { question: q } : { question: q, documents: selected };
      const response = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        setResult({ mode: runMode, text: await errorMessage(response), sources: [], isError: true });
      } else {
        const data = await response.json();
        setResult({
          mode: runMode,
          text: data.answer || "No answer was returned.",
          sources: (runMode === "ask" ? data.sources : data.documents_compared) || [],
          isError: false,
        });
      }
    } catch {
      setResult({ mode: runMode, text: "Could not reach the server. Check that the backend is running.", sources: [], isError: true });
    }
    setLoading(false);
  };

  const pickExample = (text) => {
    setQuestion(text);
    run(text);
  };

  const onSpineClick = (doc) => {
    if (mode === "compare") {
      setSelected((prev) => (prev.includes(doc) ? prev.filter((d) => d !== doc) : [...prev, doc]));
    } else {
      setQuestion(`Summarize the main contribution of ${prettyName(doc)}.`);
      questionInput.current?.focus();
    }
  };

  // The bookcase turns gently toward the mouse (skipped when the user prefers reduced motion).
  const onShelfMove = (e) => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const box = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - box.left) / box.width - 0.5;
    const y = (e.clientY - box.top) / box.height - 0.5;
    setTilt({ rx: (10 - y * 6).toFixed(2), ry: (x * 10).toFixed(2) });
  };

  const uploadFile = async (file) => {
    if (!file) return;
    setUploadStatus({ text: `Adding ${file.name} to the shelf…`, kind: "" });
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API}/upload`, { method: "POST", body: formData });
      if (!response.ok) {
        setUploadStatus({ text: `${file.name}: ${await errorMessage(response)}`, kind: "err" });
      } else {
        const data = await response.json();
        setUploadStatus({ text: `Added ${prettyName(data.filename)} (${data.chunks_stored} passages indexed).`, kind: "ok" });
        await loadDocuments();
      }
    } catch {
      setUploadStatus({ text: "Upload failed. Check that the backend is running.", kind: "err" });
    }
  };

  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const [mainText, notes] = result && !result.isError ? splitNotes(result.text) : [result?.text || "", ""];
  const isCompareResult = result?.mode === "compare";

  let caption;
  if (hovered) {
    caption = (
      <>
        <strong>{prettyName(hovered)}</strong>
        {mode === "ask" ? ". Click to ask about it." : selected.includes(hovered) ? ". Picked for comparison. Click to put it back." : ". Click to pick it for comparison."}
      </>
    );
  } else if (cited.size > 0) {
    caption = `Highlighted on the shelf: the ${cited.size === 1 ? "paper" : `${cited.size} papers`} this answer came from.`;
  } else {
    caption = "Hover over a book to see its title.";
  }

  let index = 0;

  return (
    <>
      <style>{css}</style>
      <div className={`page ${mode === "compare" ? "compare-mode" : ""}`}>
        <header className="masthead">
          <div>
            <h1>Shelfwise</h1>
            <p>Your research papers, on one shelf. Ask a question and see which papers the answer came from, or line up many papers in a single table.</p>
          </div>
          <a href={REPO_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
        </header>

        <section className="shelf-wrap" aria-label="Paper shelf">
          <div className="shelf-bar">
            <p className="hint">
              {mode === "ask"
                ? `Ask searches all ${documents.length} papers. Click a book to ask about it.`
                : `${selected.length} of ${documents.length} papers picked for comparison. Click a book to pick or drop it.`}
            </p>
            <div className="shelf-tools">
              <input
                className="filter"
                type="search"
                placeholder="Find a paper"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                aria-label="Find a paper"
              />
              {mode === "compare" && (
                <>
                  <button className="text-btn" onClick={() => setSelected(documents)}>Pick all</button>
                  <button className="text-btn" onClick={() => setSelected([])}>Clear</button>
                </>
              )}
            </div>
          </div>

          <div
            className={`shelf ${intro ? "intro" : ""} ${loading ? "scanning" : ""} ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); uploadFile(e.dataTransfer.files[0]); }}
            onMouseMove={onShelfMove}
            onMouseLeave={() => setTilt(null)}
          >
            <div className="bookcase" style={tilt ? { "--rx": `${tilt.rx}deg`, "--ry": `${tilt.ry}deg` } : undefined}>
            {documents.length === 0 && <p className="caption">The shelf is empty. Add a PDF to start.</p>}
            {sections.map((section) => (
              <div className="section" key={section.id}>
                <div className="spines">
                  {section.docs.map((doc) => {
                    const i = index++;
                    const classes = [
                      "spine",
                      selected.includes(doc) ? "on" : "off",
                      term && !doc.toLowerCase().includes(term) ? "dim" : "",
                      cited.has(doc) ? "cited" : "",
                    ].join(" ");
                    return (
                      <button
                        key={doc}
                        className={classes}
                        style={{ "--h": `${spineHeight(doc)}px`, "--spine": section.color, "--i": i }}
                        onClick={() => onSpineClick(doc)}
                        onMouseEnter={() => setHovered(doc)}
                        onMouseLeave={() => setHovered(null)}
                        onFocus={() => setHovered(doc)}
                        onBlur={() => setHovered(null)}
                        aria-label={prettyName(doc)}
                        aria-pressed={mode === "compare" ? selected.includes(doc) : undefined}
                        title={prettyName(doc)}
                      >
                        <span>{prettyName(doc)}</span>
                      </button>
                    );
                  })}
                  {section.id === sections[sections.length - 1].id && (
                    <button className="add-spine" onClick={() => fileInput.current?.click()} aria-label="Add a PDF to the shelf" title="Add a PDF">
                      +
                    </button>
                  )}
                </div>
                <span className="section-name">{section.name}</span>
              </div>
            ))}
            {documents.length === 0 && (
              <button className="add-spine" onClick={() => fileInput.current?.click()} aria-label="Add a PDF to the shelf">+</button>
            )}
            </div>
          </div>
          <input
            ref={fileInput}
            type="file"
            accept=".pdf"
            hidden
            onChange={(e) => { uploadFile(e.target.files[0]); e.target.value = ""; }}
          />
          <p className="caption" aria-live="polite">{caption}</p>
          {uploadStatus.text && <p className={`status ${uploadStatus.kind}`}>{uploadStatus.text}</p>}
        </section>

        <main className="desk">
          <div className="modes" role="group" aria-label="Mode">
            <button aria-pressed={mode === "ask"} onClick={() => setMode("ask")}>Ask</button>
            <button aria-pressed={mode === "compare"} onClick={() => setMode("compare")}>Compare</button>
          </div>
          <p className="mode-note">
            {mode === "ask"
              ? "Get an answer from the papers, with the sources it came from."
              : "Get one table comparing the picked papers. Takes about 10–20 seconds."}
          </p>

          <div className="ask-row">
            <input
              ref={questionInput}
              className="question"
              type="text"
              value={question}
              maxLength={MAX_QUESTION_LENGTH}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
              placeholder={mode === "ask" ? "Ask a question about the papers" : "What should the papers be compared on?"}
              aria-label="Question"
            />
            <button className="primary" onClick={() => run()} disabled={loading}>
              {loading ? "Reading…" : mode === "ask" ? "Ask" : "Compare"}
            </button>
          </div>

          <div className="examples">
            <span>Try:</span>
            {EXAMPLES[mode].map((ex) => (
              <button key={ex} className="example" onClick={() => pickExample(ex)} disabled={loading}>
                {ex}
              </button>
            ))}
          </div>

          {loading && (
            <section className="result" aria-live="polite">
              <p className="loading">
                {mode === "ask"
                  ? `Pulling the most relevant passages from ${documents.length} papers…`
                  : `Reading the abstracts of ${selected.length} papers, then extracting facts…`}
              </p>
            </section>
          )}

          {!loading && result && (
            <section className="result fresh" aria-live="polite">
              <div className={`result-grid ${isCompareResult ? "wide" : ""}`}>
                <div className={`answer ${result.isError ? "error" : ""}`}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{mainText}</ReactMarkdown>
                </div>
                {result.sources.length > 0 && (
                  <div className="margin">
                    <h3>{isCompareResult ? "Papers in the table" : "Sources"}</h3>
                    <ul>
                      {result.sources.map((s, i) => (
                        <li key={s}><mark style={{ "--i": i }}>{prettyName(s)}</mark></li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {notes && (
                <details className="notes">
                  <summary>Why some papers aren't in the table</summary>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{notes}</ReactMarkdown>
                </details>
              )}

              {!result.isError && (
                <div className="result-foot">
                  <button className="text-btn" onClick={copyAnswer}>{copied ? "Copied" : "Copy answer"}</button>
                </div>
              )}
            </section>
          )}
        </main>
      </div>
    </>
  );
}

export default App;
