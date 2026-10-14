import { useState, useEffect, useMemo, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Backend address: set VITE_API_URL on Vercel; falls back to the local backend.
const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
// The public demo can switch uploads off with VITE_ALLOW_UPLOADS=0.
const UPLOADS_ON = import.meta.env.VITE_ALLOW_UPLOADS !== "0";
const MAX_QUESTION_LENGTH = 2000;
const REPO_URL = "https://github.com/Prerit-kumar/ai-study-assistant";

// Shelf sections for the demo library. Uploaded papers that aren't listed go to "Your uploads".
const TOPICS = [
  { id: "attention", name: "Attention & long context", color: "var(--c-attention)",
    papers: ["Attention_Is_All_You_Need", "Transformer_XL", "Longformer", "BigBird", "Linformer", "Performer", "Reformer", "FlashAttention", "Mamba"] },
  { id: "encoders", name: "Pretrained encoders", color: "var(--c-encoders)",
    papers: ["BERT", "RoBERTa", "ALBERT", "DistilBERT", "ELECTRA", "Sentence_BERT", "T5"] },
  { id: "llms", name: "LLMs & scaling", color: "var(--c-llms)",
    papers: ["GPT3_Few_Shot_Learners", "Scaling_Laws", "Chinchilla", "LLaMA", "Llama2", "Mistral_7B", "Mixtral", "Switch_Transformers"] },
  { id: "retrieval", name: "Retrieval", color: "var(--c-retrieval)",
    papers: ["DPR_Dense_Passage_Retrieval", "REALM", "RAG_Retrieval_Augmented_Generation", "ColBERT", "RETRO", "Self_RAG", "Lost_in_the_Middle"] },
  { id: "tuning", name: "Fine-tuning & alignment", color: "var(--c-tuning)",
    papers: ["Adapters_Parameter_Efficient", "Prefix_Tuning", "LoRA", "QLoRA", "InstructGPT", "DPO"] },
  { id: "reasoning", name: "Reasoning & tools", color: "var(--c-reasoning)",
    papers: ["Chain_of_Thought", "ReAct", "Toolformer"] },
];
const UPLOADS = { id: "uploads", name: "My papers", color: "var(--c-uploads)" };
// Every paper in the demo library. Anything else in the database is one of the user's own uploads.
const LIBRARY_DOCS = new Set(TOPICS.flatMap((t) => t.papers.map((p) => `${p}.pdf`)));
const VIEWS = ["ask", "compare", "about"];
const viewFromHash = () => {
  const hash = window.location.hash.replace("#", "");
  return VIEWS.includes(hash) ? hash : "ask";
};

const MY_EXAMPLES = {
  ask: [
    "Summarize the main contribution of this paper.",
    "What method does it propose?",
    "What are the main results?",
  ],
  compare: [
    "Compare the methods proposed in these papers.",
    "Compare the datasets and results reported in these papers.",
  ],
};

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

// Shown above the two answers when both models are asked (verdicts come from the backend).
const VERDICTS = {
  agree: { kind: "good", icon: "✓", title: "Both models agree", text: "the fine-tuned model gave the same answer from the same passage" },
  differ: { kind: "info", icon: "≈", title: "Both models answered", text: "but from different passages, so check the sources" },
  warning: { kind: "warn", icon: "!", title: "Not confirmed", text: "the fine-tuned model couldn't find this in the passages, so check the sources before relying on it" },
  local_only: { kind: "info", icon: "↑", title: "Answered by the fine-tuned model", text: "GPT couldn't find it in the passages" },
  not_found: { kind: "good", icon: "✓", title: "Both models agree", text: "this isn't in the papers" },
  local_unavailable: { kind: "warn", icon: "!", title: "GPT only", text: "the fine-tuned model isn't running" },
};

const css = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400;1,9..144,600&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  color-scheme: dark;
  --bg: #090c14;
  --bg-2: #0e1320;
  --panel: rgba(20, 26, 42, 0.72);
  --panel-solid: #141a2a;
  --panel-hi: rgba(255, 255, 255, 0.04);
  --line: rgba(255, 255, 255, 0.08);
  --line-strong: rgba(255, 255, 255, 0.16);
  --ink: #ece7dc;
  --ink-soft: #c4c0b6;
  --muted: #8a90a2;
  --gold: #d9b26f;
  --gold-hi: #f2d49b;
  --gold-deep: #9c7638;
  --rose: #e48a7a;
  --violet: #8f87f1;
  --danger: #f2a49e;
  --ok: #8fd4a8;
  --wood: #2a1d14;
  --wood-top: #3a2a1d;
  --wood-edge: #1a120c;
  --pages: #e9dfc7;
  --pages-line: #b9ab8c;
  --foil: rgba(242, 212, 155, 0.75);
  --depth: 60px;
  --c-attention: #1f3b66;
  --c-encoders: #6e1f2e;
  --c-llms: #1d4f3d;
  --c-retrieval: #7c561b;
  --c-tuning: #45306a;
  --c-reasoning: #15525c;
  --c-uploads: #3b4150;
  --serif: "Fraunces", Georgia, serif;
  --sans: "Manrope", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, monospace;
  --ease: cubic-bezier(.2,.7,.2,1);
}

* { box-sizing: border-box; }
html, body { background: var(--bg); }
body { margin: 0; color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.55; -webkit-font-smoothing: antialiased; }
button, input { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--gold); outline-offset: 3px; }
::selection { background: rgba(217, 178, 111, 0.35); }

/* ---------- ambient background ---------- */
.ambient { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.ambient::before, .ambient::after {
  content: ""; position: absolute; width: 720px; height: 720px; border-radius: 50%; filter: blur(110px); opacity: 0.22;
}
.ambient::before { background: #3b3fa8; top: -320px; left: -180px; }
.ambient::after { background: #8a5a22; bottom: -380px; right: -200px; opacity: 0.18; }
.grain {
  position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.5;
  background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, #000 20%, transparent 75%);
  -webkit-mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, #000 20%, transparent 75%);
}
@media (prefers-reduced-motion: no-preference) {
  .ambient::before { animation: drift 38s ease-in-out infinite alternate; }
  .ambient::after { animation: drift 46s ease-in-out infinite alternate-reverse; }
  @keyframes drift { to { transform: translate(140px, 90px) scale(1.12); } }
}

.page { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 36px 24px 80px; }
@media (max-width: 640px) { .page { padding: 22px 16px 60px; } }

/* ---------- header ---------- */
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.brandmark { display: inline-flex; align-items: center; gap: 10px; font-family: var(--mono); font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.brandmark i {
  width: 22px; height: 22px; border-radius: 6px; display: inline-block;
  background: linear-gradient(135deg, var(--gold-hi), var(--gold-deep));
  box-shadow: 0 0 0 1px rgba(255,255,255,0.15) inset, 0 6px 18px -6px rgba(217,178,111,0.7);
  position: relative;
}
.brandmark i::after { content: ""; position: absolute; left: 6px; right: 6px; top: 5px; bottom: 5px; border-left: 2px solid #2a1d0c; border-right: 2px solid #2a1d0c; }
.gh {
  display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px; border: 1px solid var(--line-strong);
  color: var(--ink-soft); text-decoration: none; font-size: 13px; font-weight: 600; background: var(--panel-hi);
  transition: border-color 200ms ease, color 200ms ease, background 200ms ease;
}
.gh:hover { border-color: var(--gold); color: var(--gold-hi); background: rgba(217,178,111,0.08); }
.topnav { display: flex; align-items: center; gap: 18px; }
.navlink { color: var(--ink-soft); text-decoration: none; font-size: 13px; font-weight: 600; }
.navlink:hover { color: var(--gold-hi); }
html { scroll-behavior: smooth; }
.gh svg { width: 16px; height: 16px; fill: currentColor; }

.hero { margin-top: 44px; max-width: 820px; }
.eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--gold); margin: 0 0 16px; display: flex; gap: 10px; align-items: center; }
.eyebrow::before { content: ""; width: 28px; height: 1px; background: var(--gold); }
.hero h1 { font-family: var(--serif); font-weight: 400; font-size: clamp(42px, 6.4vw, 76px); line-height: 1.02; letter-spacing: -0.025em; margin: 0; }
.hero h1 em {
  font-style: italic; font-weight: 600;
  background: linear-gradient(100deg, var(--gold-hi) 0%, var(--gold) 40%, var(--rose) 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 18px 0 0; color: var(--ink-soft); font-size: 17px; max-width: 60ch; }
.stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
.stat { display: inline-flex; align-items: baseline; gap: 7px; padding: 7px 13px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel-hi); font-size: 13px; color: var(--muted); }
.stat b { font-family: var(--mono); font-weight: 500; color: var(--ink); }

/* ---------- panels ---------- */
.panel {
  background: linear-gradient(180deg, rgba(255,255,255,0.035), transparent 40%), var(--panel);
  border: 1px solid var(--line); border-radius: 18px;
  box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 30px 60px -30px rgba(0,0,0,0.8);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}

/* ---------- bookshelf ---------- */
.shelf-wrap { margin-top: 40px; padding: 18px 22px 16px; }
.shelf-bar { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; justify-content: space-between; }
.label { font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.shelf-bar .hint { color: var(--ink-soft); font-size: 14px; margin: 4px 0 0; }
.shelf-tools { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.filter {
  width: 210px; padding: 8px 12px 8px 32px; border: 1px solid var(--line); border-radius: 10px; background: rgba(0,0,0,0.25);
  background-image: radial-gradient(circle at 13px 50%, transparent 4px, var(--muted) 4.5px, var(--muted) 5.5px, transparent 6px);
  transition: border-color 200ms ease;
}
.filter::placeholder { color: var(--muted); }
.filter:focus { border-color: var(--gold); outline: none; }
.text-btn { background: none; border: 0; padding: 0; color: var(--gold); cursor: pointer; font-size: 13px; font-weight: 600; }
.text-btn:hover { color: var(--gold-hi); }

.shelf { perspective: 1400px; perspective-origin: 50% -30%; padding: 34px 26px 34px; }
.shelf.dragging { outline: 1px dashed var(--gold); outline-offset: 4px; border-radius: 12px; background: rgba(217,178,111,0.05); }
.bookcase {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 48px 34px; transform-style: preserve-3d;
  transform: rotateX(var(--rx, 9deg)) rotateY(var(--ry, 0deg));
  transition: transform 600ms var(--ease);
}
@media (max-width: 640px) { .shelf { overflow-x: auto; } .bookcase { flex-wrap: nowrap; justify-content: flex-start; } }
.section { flex: 0 0 auto; display: flex; flex-direction: column; transform-style: preserve-3d; }
.spines { position: relative; display: flex; align-items: flex-end; gap: 3px; height: 160px; padding: 0 10px; transform-style: preserve-3d; }
.spines::before {
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: var(--depth);
  transform-origin: bottom; transform: rotateX(90deg);
  background: linear-gradient(180deg, rgba(0,0,0,0.35), transparent), var(--wood-top);
}
.spines::after {
  content: ""; position: absolute; left: 0; right: 0; top: 100%; height: 12px;
  background: linear-gradient(var(--wood-top), var(--wood-edge));
  border-top: 1px solid rgba(217,178,111,0.45); border-radius: 0 0 3px 3px;
  box-shadow: 0 14px 22px -8px rgba(0,0,0,0.8);
}
.section-name { font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; color: var(--muted); margin-top: 22px; white-space: nowrap; }
.section-name::before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 2px; background: var(--dot); margin-right: 8px; vertical-align: 1px; box-shadow: 0 0 0 1px rgba(255,255,255,0.18) inset; }

.spine {
  --book: var(--spine);
  position: relative; width: 27px; height: var(--h); border: 0; padding: 0; cursor: pointer; color: #f1e6cc;
  border-radius: 2px 2px 0 0; transform-style: preserve-3d; transform-origin: bottom center;
  background:
    linear-gradient(var(--foil), var(--foil)) no-repeat 50% 10px / calc(100% - 8px) 1.5px,
    linear-gradient(var(--foil), var(--foil)) no-repeat 50% 14px / calc(100% - 8px) 1px,
    linear-gradient(var(--foil), var(--foil)) no-repeat 50% calc(100% - 12px) / calc(100% - 8px) 1.5px,
    linear-gradient(90deg, rgba(255,255,255,0.14), transparent 30%, rgba(0,0,0,0.35)),
    var(--book);
  box-shadow: 0 0 0 1px rgba(0,0,0,0.35) inset;
  transition: transform 320ms var(--ease), box-shadow 320ms ease, color 200ms ease, filter 200ms ease;
}
.spine::before {
  content: ""; position: absolute; left: 0; top: 0; width: 100%; height: var(--depth);
  transform-origin: top; transform: rotateX(-90deg);
  background: repeating-linear-gradient(90deg, var(--pages) 0 2px, var(--pages-line) 2px 3px);
  box-shadow: inset 0 0 0 2px var(--book);
}
.spine::after {
  content: ""; position: absolute; top: 0; left: 100%; width: var(--depth); height: 100%;
  transform-origin: left; transform: rotateY(90deg);
  background: linear-gradient(90deg, rgba(0,0,0,0.5), rgba(0,0,0,0.2)), var(--book);
}
.spine span {
  position: absolute; inset: 20px 0 18px 0; writing-mode: vertical-rl; transform: rotate(180deg);
  font-family: var(--serif); font-size: 11.5px; font-weight: 600; letter-spacing: 0.03em; line-height: 27px; text-align: left;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.spine:hover, .spine:focus-visible { transform: translateZ(40px) translateY(-8px) rotateY(-14deg); filter: brightness(1.18); z-index: 2; }
.compare-mode .spine.off { --book: color-mix(in srgb, var(--spine) 38%, #0d1220); color: rgba(241,230,204,0.45); }
.compare-mode .spine.on { transform: translateZ(10px); box-shadow: 0 0 0 1.5px var(--gold-hi), 0 0 16px rgba(217,178,111,0.35); }
.compare-mode .spine.on:hover, .compare-mode .spine.off:hover { transform: translateZ(40px) translateY(-8px) rotateY(-14deg); }
.page .spine.dim { --book: color-mix(in srgb, var(--spine) 18%, #0d1220); color: transparent; }
.page .spine.cited {
  transform: translateZ(48px) translateY(-10px) rotateY(-10deg);
  box-shadow: 0 0 0 2px var(--gold-hi), 0 0 28px 4px rgba(217,178,111,0.45), 0 18px 28px rgba(0,0,0,0.6); z-index: 3;
}

.add-spine {
  width: 27px; height: 128px; border: 1px dashed var(--line-strong); border-radius: 2px 2px 0 0; background: rgba(255,255,255,0.02);
  color: var(--muted); font-size: 18px; cursor: pointer; align-self: flex-end; transition: border-color 200ms ease, color 200ms ease;
}
.add-spine:hover { border-color: var(--gold); color: var(--gold); }

.caption { min-height: 22px; margin: 16px 0 0; font-size: 14px; color: var(--muted); }
.caption strong { color: var(--gold-hi); font-family: var(--serif); font-weight: 600; font-size: 15px; }
.status { font-size: 14px; margin: 4px 0 0; }
.status.ok { color: var(--ok); }
.status.err { color: var(--danger); }

@media (prefers-reduced-motion: no-preference) {
  .shelf.intro .spine { animation: settle 700ms var(--ease) backwards; animation-delay: calc(200ms + var(--i) * 18ms); }
  @keyframes settle { from { opacity: 0; transform: translateZ(26px) translateY(-14px); } }
  .reveal { animation: reveal 800ms var(--ease) backwards; animation-delay: var(--d, 0ms); }
  @keyframes reveal { from { opacity: 0; transform: translateY(14px); filter: blur(6px); } }
  .shelf.scanning .spine { animation: glow 1.6s ease-in-out infinite; animation-delay: calc(var(--i) * 40ms); }
  @keyframes glow { 0%, 70%, 100% { filter: none; } 35% { filter: brightness(1.6) saturate(1.2); } }
}
@media (prefers-reduced-motion: reduce) { .bookcase, .spine { transition: none; } }

/* ---------- desk ---------- */
.desk { margin-top: 40px; max-width: 940px; }
.modes { position: relative; display: inline-grid; grid-template-columns: 1fr 1fr; padding: 4px; border-radius: 12px; border: 1px solid var(--line); background: rgba(0,0,0,0.3); }
.modes::before {
  content: ""; position: absolute; top: 4px; bottom: 4px; left: 4px; width: calc(50% - 4px); border-radius: 9px;
  background: linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.04)); border: 1px solid var(--line-strong);
  box-shadow: 0 6px 16px -8px rgba(0,0,0,0.8);
  transition: transform 380ms var(--ease);
}
.modes.compare::before { transform: translateX(100%); }
.modes button { position: relative; border: 0; background: none; padding: 9px 30px; cursor: pointer; font-weight: 600; font-size: 14px; color: var(--muted); transition: color 200ms ease; }
.modes button[aria-pressed="true"] { color: var(--ink); }
.mode-note { color: var(--muted); margin: 12px 0 16px; font-size: 14px; }

.ask-row {
  display: flex; gap: 8px; padding: 6px; border-radius: 16px; border: 1px solid var(--line-strong);
  background: rgba(8, 11, 19, 0.7); transition: border-color 250ms ease, box-shadow 250ms ease;
}
.ask-row:focus-within { border-color: rgba(217,178,111,0.6); box-shadow: 0 0 0 4px rgba(217,178,111,0.1), 0 20px 50px -24px rgba(217,178,111,0.35); }
.question { flex: 1; min-width: 0; padding: 13px 14px; font-size: 16px; border: 0; background: none; outline: none; }
.question::placeholder { color: var(--muted); }
.primary {
  padding: 0 26px; border: 0; border-radius: 11px; color: #1d1408; font-weight: 700; cursor: pointer; white-space: nowrap;
  background: linear-gradient(180deg, var(--gold-hi), var(--gold) 55%, var(--gold-deep));
  box-shadow: 0 1px 0 rgba(255,255,255,0.4) inset, 0 10px 24px -10px rgba(217,178,111,0.7);
  transition: transform 160ms ease, box-shadow 200ms ease, filter 200ms ease;
}
.primary:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.06); box-shadow: 0 1px 0 rgba(255,255,255,0.4) inset, 0 14px 30px -10px rgba(217,178,111,0.85); }
.primary:active:not(:disabled) { transform: translateY(1px); }
.primary:disabled { opacity: 0.6; cursor: progress; }
@media (max-width: 520px) { .ask-row { flex-direction: column; } .primary { padding: 12px; } }

.examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; align-items: center; }
.examples .label { margin-right: 4px; }
.example {
  border: 1px solid var(--line); background: var(--panel-hi); border-radius: 999px; padding: 7px 14px; font-size: 13px; cursor: pointer; text-align: left; color: var(--ink-soft);
  transition: border-color 200ms ease, color 200ms ease, background 200ms ease, transform 200ms ease;
}
.example:hover:not(:disabled) { border-color: rgba(217,178,111,0.55); color: var(--gold-hi); background: rgba(217,178,111,0.07); transform: translateY(-1px); }

/* ---------- result ---------- */
.result { margin-top: 30px; padding: 26px 28px; }
@media (max-width: 640px) { .result { padding: 20px 18px; } }
@media (prefers-reduced-motion: no-preference) {
  .result.fresh { animation: reveal 600ms var(--ease); }
}
.result-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.result-grid { display: grid; grid-template-columns: minmax(0, 1fr) 210px; gap: 32px; }
.result-grid.wide { grid-template-columns: 1fr; }
@media (max-width: 900px) { .result-grid { grid-template-columns: 1fr; } }

.answer { font-family: var(--serif); font-size: 18px; line-height: 1.7; color: var(--ink); overflow-x: auto; }
.answer p { margin: 0 0 0.9em; max-width: 70ch; }
.answer strong { color: var(--gold-hi); font-weight: 600; }
.answer code { font-family: var(--mono); font-size: 0.85em; padding: 1px 5px; border-radius: 4px; background: rgba(255,255,255,0.06); }
.answer table { border-collapse: collapse; width: 100%; margin: 4px 0 18px; font-family: var(--sans); font-size: 14px; line-height: 1.5; }
.answer th, .answer td { border-bottom: 1px solid var(--line); padding: 11px 12px; text-align: left; vertical-align: top; }
.answer th { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: var(--gold); border-bottom: 1px solid rgba(217,178,111,0.4); }
.answer tbody tr { transition: background 160ms ease; }
.answer tbody tr:hover { background: rgba(255,255,255,0.03); }
.answer td:first-child { font-weight: 600; color: var(--ink); }
.answer.error { font-family: var(--sans); font-size: 15px; color: var(--danger); }

.loading { display: flex; flex-direction: column; gap: 12px; }
.loading p { margin: 0 0 6px; font-family: var(--serif); font-style: italic; font-size: 17px; color: var(--ink-soft); }
.bar { height: 11px; border-radius: 6px; background: linear-gradient(90deg, rgba(255,255,255,0.04) 0%, rgba(217,178,111,0.18) 50%, rgba(255,255,255,0.04) 100%) 0 0 / 200% 100%; }
.bar:nth-child(3) { width: 86%; } .bar:nth-child(4) { width: 64%; }
@media (prefers-reduced-motion: no-preference) {
  .bar { animation: shimmer 1.6s linear infinite; }
  @keyframes shimmer { to { background-position: -200% 0; } }
}

.margin { border-left: 1px solid var(--line); padding-left: 22px; }
.result-grid.wide .margin { border-left: 0; padding-left: 0; border-top: 1px solid var(--line); padding-top: 16px; }
@media (max-width: 900px) { .margin { border-left: 0; padding-left: 0; border-top: 1px solid var(--line); padding-top: 16px; } }
.margin .label { display: block; margin-bottom: 12px; }
.margin ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; counter-reset: src; }
.result-grid.wide .margin ol { flex-direction: row; flex-wrap: wrap; gap: 8px; }
.margin li {
  counter-increment: src; display: flex; gap: 10px; align-items: baseline; font-size: 14px; color: var(--ink); overflow-wrap: anywhere;
}
.margin li::before { content: counter(src, decimal-leading-zero); flex-shrink: 0; font-family: var(--mono); font-size: 11px; color: var(--gold); }
.result-grid.wide .margin li { padding: 5px 12px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel-hi); }
@media (prefers-reduced-motion: no-preference) {
  .result.fresh .margin li { animation: reveal 500ms var(--ease) backwards; animation-delay: calc(250ms + var(--i) * 90ms); }
}

.notes { margin-top: 20px; border-top: 1px solid var(--line); padding-top: 14px; font-size: 14px; color: var(--muted); }
.notes summary { cursor: pointer; color: var(--ink-soft); font-weight: 600; }
.notes summary:hover { color: var(--gold-hi); }
.notes p { margin: 8px 0 0; }
.engine { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 0 0 14px; }
.engine-opts { display: inline-flex; flex-wrap: wrap; gap: 6px; }
.engine-opts button {
  border: 1px solid var(--line); background: var(--panel-hi); color: var(--ink-soft); border-radius: 999px; padding: 6px 14px;
  font-size: 13px; font-weight: 600; cursor: pointer; transition: border-color 200ms ease, color 200ms ease, background 200ms ease;
}
.engine-opts button:hover:not(:disabled) { border-color: rgba(217,178,111,0.45); }
.engine-opts button[aria-pressed="true"] { border-color: rgba(217,178,111,0.6); color: var(--gold-hi); background: rgba(217,178,111,0.1); }
.engine-opts button:disabled { opacity: 0.45; cursor: not-allowed; }
.engine-note { font-size: 12px; color: var(--muted); display: inline-flex; align-items: center; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 7px; background: var(--muted); }
.dot.on { background: var(--ok); box-shadow: 0 0 8px var(--ok); }

.check { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
.check-line { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); }
.check-line b { font-weight: 600; }
.check-icon { width: 20px; height: 20px; border-radius: 6px; flex-shrink: 0; display: grid; place-items: center; font-size: 12px; font-weight: 700; }
.check-line.good b { color: var(--ok); }
.check-line.good .check-icon { background: rgba(143,212,168,0.16); color: var(--ok); }
.check-line.warn b { color: var(--danger); }
.check-line.warn .check-icon { background: rgba(242,164,158,0.16); color: var(--danger); }
.check-line.info b { color: var(--gold-hi); }
.check-line.info .check-icon { background: rgba(217,178,111,0.16); color: var(--gold-hi); }
.check-more { margin-top: 10px; }
.check-more summary { cursor: pointer; font-size: 13px; font-weight: 600; color: var(--ink-soft); width: fit-content; }
.check-more summary:hover { color: var(--gold-hi); }
.check-more .duo { margin-top: 14px; }
.duo { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 860px) { .duo { grid-template-columns: 1fr; } }
.duo-card { border: 1px solid var(--line); border-radius: 14px; padding: 18px 18px 16px; background: rgba(0,0,0,0.18); min-width: 0; }
.duo-card.best { border-color: rgba(217,178,111,0.55); box-shadow: 0 0 0 1px rgba(217,178,111,0.2), 0 18px 40px -24px rgba(217,178,111,0.5); }
.duo-card.flagged { border-color: rgba(242,164,158,0.45); }
.duo-top { display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; margin-bottom: 12px; }
.duo-name { font-weight: 700; font-size: 14px; }
.duo-sub { display: block; font-family: var(--mono); font-size: 11px; color: var(--muted); margin-top: 3px; }
.tag { font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; padding: 4px 9px; border-radius: 999px; white-space: nowrap; }
.tag.best { color: #1d1408; background: linear-gradient(180deg, var(--gold-hi), var(--gold)); font-weight: 600; }
.tag.flag { color: var(--danger); border: 1px solid rgba(242,164,158,0.5); }
.duo .answer { font-size: 15px; line-height: 1.6; }
.duo-sources { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.duo-sources span { font-size: 12px; padding: 3px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--ink-soft); }
.duo-empty { color: var(--muted); font-size: 14px; font-style: italic; margin: 0; }

.copy { border: 1px solid var(--line); background: var(--panel-hi); color: var(--ink-soft); border-radius: 8px; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer; transition: border-color 200ms ease, color 200ms ease; }
.copy:hover { border-color: var(--gold); color: var(--gold-hi); }

/* ---------- about ---------- */
.about { margin-top: 96px; scroll-margin-top: 24px; }
.about-head { max-width: 760px; }
.about h2 { font-family: var(--serif); font-weight: 400; font-size: clamp(32px, 4.4vw, 48px); line-height: 1.08; letter-spacing: -0.02em; margin: 0; }
.about h2 em { font-style: italic; color: var(--gold-hi); }
.about-head p { color: var(--ink-soft); font-size: 16px; margin: 16px 0 0; max-width: 64ch; }
.about h3 { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 52px 0 16px; }

.how { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 860px) { .how { grid-template-columns: 1fr; } }
.how-card { padding: 22px 22px 24px; transition: border-color 250ms ease, transform 250ms var(--ease); }
.how-card:hover { border-color: rgba(217,178,111,0.35); transform: translateY(-3px); }
.how-card .num { font-family: var(--mono); font-size: 12px; color: var(--gold); }
.how-card h4 { font-family: var(--serif); font-weight: 600; font-size: 21px; margin: 10px 0 8px; }
.how-card p { margin: 0; color: var(--ink-soft); font-size: 14px; }
.how-card .flow { margin-top: 14px; font-family: var(--mono); font-size: 11px; color: var(--muted); line-height: 1.7; }

.results { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 860px) { .results { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 420px) { .results { grid-template-columns: 1fr; } }
.metric { padding: 20px 20px 22px; }
.metric b {
  display: block; font-family: var(--serif); font-weight: 600; font-size: 40px; line-height: 1; letter-spacing: -0.02em;
  background: linear-gradient(100deg, var(--gold-hi), var(--gold) 50%, var(--rose)); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.metric span { display: block; margin-top: 10px; font-size: 14px; color: var(--ink); font-weight: 600; }
.metric small { display: block; margin-top: 4px; font-size: 13px; color: var(--muted); }

.stack { display: flex; flex-wrap: wrap; gap: 8px; }
.stack span { padding: 7px 13px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel-hi); font-family: var(--mono); font-size: 12px; color: var(--ink-soft); }

.builder { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 18px; padding: 24px 26px; }
.builder-who { display: flex; align-items: center; gap: 16px; }
.avatar {
  width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center; flex-shrink: 0;
  font-family: var(--serif); font-weight: 600; font-size: 20px; color: #1d1408;
  background: linear-gradient(135deg, var(--gold-hi), var(--gold-deep));
  box-shadow: 0 10px 24px -10px rgba(217,178,111,0.7);
}
.builder-who strong { display: block; font-family: var(--serif); font-size: 20px; font-weight: 600; }
.builder-who p { margin: 2px 0 0; color: var(--muted); font-size: 14px; }

.footer { margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px 20px; justify-content: space-between; color: var(--muted); font-size: 13px; }
.footer a { color: var(--muted); text-decoration: none; }
.footer a:hover { color: var(--gold-hi); }

/* ---------- navigation ---------- */
.topbar {
  display: block; position: sticky; top: 0; z-index: 20;
  background: rgba(9, 12, 20, 0.78); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--line);
}
.topbar-inner { max-width: 1200px; margin: 0 auto; padding: 14px 24px; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
@media (max-width: 640px) { .topbar-inner { padding: 12px 16px; } .brandmark span, .gh span { display: none; } }
.nav { display: inline-flex; gap: 2px; padding: 4px; border: 1px solid var(--line); border-radius: 12px; background: rgba(0,0,0,0.3); }
.nav a { padding: 7px 16px; border-radius: 9px; font-size: 14px; font-weight: 600; color: var(--muted); text-decoration: none; transition: color 200ms ease, background 200ms ease; }
@media (max-width: 420px) { .nav a { padding: 7px 11px; } }
.nav a:hover { color: var(--ink); }
.nav a[aria-current="page"] { color: var(--ink); background: linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.04)); box-shadow: inset 0 0 0 1px var(--line-strong); }

.intro-head { margin-top: 8px; }
.intro-head h1 { font-family: var(--serif); font-weight: 400; font-size: clamp(34px, 4.6vw, 52px); line-height: 1.05; letter-spacing: -0.02em; margin: 0; }
.intro-head h1 em { font-style: italic; color: var(--gold-hi); }
.intro-head p { margin: 12px 0 0; color: var(--ink-soft); font-size: 16px; max-width: 60ch; }
.view-about .about { margin-top: 8px; }

/* ---------- library / my papers ---------- */
.scope { display: inline-flex; gap: 6px; flex-wrap: wrap; }
.scope button {
  display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line); background: var(--panel-hi); color: var(--ink-soft);
  border-radius: 999px; padding: 6px 14px; font-size: 13px; font-weight: 600; cursor: pointer; transition: border-color 200ms ease, color 200ms ease, background 200ms ease;
}
.scope button span { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.scope button[aria-pressed="true"] { border-color: rgba(217,178,111,0.6); color: var(--gold-hi); background: rgba(217,178,111,0.1); }
.scope button[aria-pressed="true"] span { color: var(--gold); }
.upload-btn { border: 1px solid var(--line-strong); background: var(--panel-hi); color: var(--ink-soft); border-radius: 10px; padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer; transition: border-color 200ms ease, color 200ms ease; }
.upload-btn:hover { border-color: var(--gold); color: var(--gold-hi); }
.empty-shelf {
  display: block; width: 100%; margin: 22px 0 6px; padding: 48px 20px; border: 1px dashed var(--line-strong); border-radius: 14px;
  background: rgba(255,255,255,0.02); color: var(--muted); font-size: 15px; text-align: center; cursor: pointer; transition: border-color 200ms ease, color 200ms ease, background 200ms ease;
}
.my-files { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 14px; }
.my-files .label { margin-right: 4px; }
.file-chip {
  display: inline-flex; align-items: center; gap: 8px; padding: 4px 5px 4px 12px; border: 1px solid var(--line); border-radius: 999px;
  background: var(--panel-hi); font-size: 13px; color: var(--ink-soft); max-width: 100%;
}
.file-chip span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-chip button {
  border: 0; background: rgba(255,255,255,0.06); color: var(--muted); border-radius: 999px; min-width: 24px; height: 24px; padding: 0 8px;
  cursor: pointer; font-size: 12px; font-weight: 700; flex-shrink: 0; transition: color 150ms ease, background 150ms ease;
}
.file-chip button:hover { color: var(--danger); background: rgba(242,164,158,0.14); }
.file-chip button.confirm { color: #2a0d0b; background: var(--danger); }
.file-chip button:disabled { opacity: 0.5; cursor: progress; }
.empty-shelf strong { display: block; font-family: var(--serif); font-size: 22px; font-weight: 400; color: var(--ink); margin-bottom: 6px; }
.empty-shelf:hover, .empty-shelf.dragging { border-color: var(--gold); color: var(--ink-soft); background: rgba(217,178,111,0.05); }
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

function GitHubIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function AnswerCard({ name, sub, part, best, flagged }) {
  return (
    <article className={`duo-card ${best ? "best" : ""} ${flagged ? "flagged" : ""}`}>
      <div className="duo-top">
        <div>
          <div className="duo-name">{name}</div>
          <span className="duo-sub">{sub}</span>
        </div>
        {best && <span className="tag best">Best supported</span>}
        {flagged && <span className="tag flag">Unverified</span>}
      </div>
      {part ? (
        <>
          <div className="answer">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{part.answer}</ReactMarkdown>
          </div>
          {part.sources.length > 0 && (
            <div className="duo-sources">
              {part.sources.map((s) => <span key={s}>{prettyName(s)}</span>)}
            </div>
          )}
        </>
      ) : (
        <p className="duo-empty">Not running. Start finetune/model_server.py to see this answer.</p>
      )}
    </article>
  );
}

function ModelCheck({ both }) {
  const verdict = VERDICTS[both.verdict] || VERDICTS.differ;
  const localSeconds = both.local?.seconds != null ? ` · ${both.local.seconds}s` : "";
  return (
    <div className="check">
      <div className={`check-line ${verdict.kind}`}>
        <span className="check-icon" aria-hidden="true">{verdict.icon}</span>
        <span><b>{verdict.title}</b> · {verdict.text}</span>
      </div>
      {both.local && (
        <details className="check-more">
          <summary>See both answers</summary>
          <div className="duo">
            <AnswerCard name="GPT-4o-mini" sub="OpenAI API" part={both.gpt} best={both.best === "gpt"} flagged={both.verdict === "warning"} />
            <AnswerCard name="Fine-tuned Llama-3.2-3B" sub={`QLoRA v2 · local GPU${localSeconds}`} part={both.local} best={both.best === "local"} flagged={false} />
          </div>
        </details>
      )}
    </div>
  );
}

function App() {
  const [view, setView] = useState(viewFromHash); // "ask" | "compare" | "about"
  const mode = view === "compare" ? "compare" : "ask";
  const [scope, setScope] = useState("library"); // "library" (demo papers) or "mine" (the user's own uploads)
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null); // { mode, text, sources, both, isError }
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ text: "", kind: "" });
  const [dragging, setDragging] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]); // papers picked for Compare (nothing is picked automatically)
  const [filter, setFilter] = useState("");
  const [hovered, setHovered] = useState(null);
  const [tilt, setTilt] = useState(null);
  const [intro, setIntro] = useState(true);
  const [engine, setEngine] = useState("gpt"); // "gpt" or "both" (GPT + fine-tuned local model)
  const [localReady, setLocalReady] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(null); // file waiting for a second click to delete
  const [deleting, setDeleting] = useState(null);
  const fileInput = useRef(null);
  const questionInput = useRef(null);

  const loadDocuments = async () => {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error(await errorMessage(res));
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch {
      setDocuments([]);
    }
  };

  useEffect(() => {
    document.title = "Study Assistant AI";
    loadDocuments();
    const timer = setTimeout(() => setIntro(false), 2400);
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);

    // Is the fine-tuned model server (finetune/model_server.py) running? Checked every 10 seconds.
    const checkLocal = async () => {
      try {
        const res = await fetch(`${API}/local-model`);
        setLocalReady(res.ok && (await res.json()).available === true);
      } catch {
        setLocalReady(false);
      }
    };
    checkLocal();
    const poll = setInterval(checkLocal, 10000);
    return () => { clearTimeout(timer); clearInterval(poll); window.removeEventListener("hashchange", onHash); };
  }, []);

  // A new page starts clean.
  useEffect(() => {
    setResult(null);
    setQuestion("");
    window.scrollTo({ top: 0 });
  }, [view]);

  const libraryDocs = useMemo(() => documents.filter((d) => LIBRARY_DOCS.has(d)), [documents]);
  const myDocs = useMemo(() => documents.filter((d) => !LIBRARY_DOCS.has(d)), [documents]);
  const scopeDocs = scope === "library" ? libraryDocs : myDocs;

  // Shelf sections: topics for the demo library, one shelf for the user's own papers.
  const sections = useMemo(() => {
    if (scope === "mine") return myDocs.length > 0 ? [{ ...UPLOADS, docs: myDocs }] : [];
    return TOPICS.map((topic) => ({
      ...topic,
      docs: topic.papers.map((p) => `${p}.pdf`).filter((d) => libraryDocs.includes(d)),
    })).filter((t) => t.docs.length > 0);
  }, [scope, libraryDocs, myDocs]);

  const switchScope = (next) => {
    setScope(next);
    setSelected([]);
    setResult(null);
    setFilter("");
  };

  const term = filter.trim().toLowerCase();
  const cited = new Set(result && !result.isError ? result.sources : []);
  const examples = (scope === "mine" ? MY_EXAMPLES : EXAMPLES)[mode];

  const run = async (text = question, picked = selected) => {
    const q = text.trim();
    if (loading || !q) return;

    if (scopeDocs.length === 0) {
      setResult({ mode, text: "Upload a PDF first. Your questions will search only your own papers.", sources: [], isError: true });
      return;
    }
    if (mode === "compare" && picked.length < 2) {
      setResult({ mode, text: "Pick at least 2 books on the shelf to compare them, or press Pick all.", sources: [], isError: true });
      return;
    }

    setLoading(true);
    setCopied(false);
    setResult(null);

    try {
      const endpoint = mode === "ask" ? "/ask" : "/compare";
      // Ask searches only the shelf that is open: the demo library or the user's own papers.
      const askDocs = scope === "mine" || myDocs.length > 0 ? scopeDocs : [];
      const body = mode === "ask" ? { question: q, mode: engine, documents: askDocs } : { question: q, documents: picked };
      const response = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        setResult({ mode, text: await errorMessage(response), sources: [], isError: true });
      } else {
        const data = await response.json();
        setResult({
          mode,
          text: data.answer || "No answer was returned.",
          sources: (mode === "ask" ? data.sources : data.documents_compared) || [],
          both: mode === "ask" ? data.both || null : null,
          isError: false,
        });
      }
    } catch {
      setResult({ mode, text: "Could not reach the server. Check that the backend is running.", sources: [], isError: true });
    }
    setLoading(false);
  };

  const pickExample = (text) => {
    setQuestion(text);
    // A Compare example with nothing picked compares every paper on the open shelf.
    if (mode === "compare" && selected.length < 2) {
      setSelected(scopeDocs);
      run(text, scopeDocs);
    } else {
      run(text);
    }
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
    setTilt({ rx: (9 - y * 5).toFixed(2), ry: (x * 8).toFixed(2) });
  };

  const uploadFile = async (file) => {
    if (!file || !UPLOADS_ON) return;
    setUploadStatus({ text: `Adding ${file.name} to your papers…`, kind: "" });
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
        // Open the user's own shelf, so questions search only their papers.
        const wasMine = scope === "mine";
        setScope("mine");
        setResult(null);
        setFilter("");
        setSelected((prev) => (wasMine ? [...new Set([...prev, data.filename])] : [data.filename]));
      }
    } catch {
      setUploadStatus({ text: "Upload failed. Check that the backend is running.", kind: "err" });
    }
  };

  // Delete one of the user's own PDFs. The first click asks "Delete?", the second click deletes.
  const deleteFile = async (doc) => {
    if (confirmDelete !== doc) {
      setConfirmDelete(doc);
      setTimeout(() => setConfirmDelete((current) => (current === doc ? null : current)), 3000);
      return;
    }
    setConfirmDelete(null);
    setDeleting(doc);
    try {
      const response = await fetch(`${API}/documents/${encodeURIComponent(doc)}`, { method: "DELETE" });
      if (!response.ok) {
        setUploadStatus({ text: `${prettyName(doc)}: ${await errorMessage(response)}`, kind: "err" });
      } else {
        setUploadStatus({ text: `Deleted ${prettyName(doc)}.`, kind: "ok" });
        setSelected((prev) => prev.filter((d) => d !== doc));
        if (result?.sources?.includes(doc)) setResult(null);
        await loadDocuments();
      }
    } catch {
      setUploadStatus({ text: "Delete failed. Check that the backend is running.", kind: "err" });
    }
    setDeleting(null);
  };

  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const dropHandlers = {
    onDragOver: (e) => { e.preventDefault(); if (UPLOADS_ON) setDragging(true); },
    onDragLeave: () => setDragging(false),
    onDrop: (e) => { e.preventDefault(); setDragging(false); uploadFile(e.dataTransfer.files[0]); },
  };

  const [mainText, notes] = result && !result.isError ? splitNotes(result.text) : [result?.text || "", ""];
  const isCompareResult = result?.mode === "compare";

  let hint;
  if (scope === "mine" && myDocs.length === 0) {
    hint = "Upload your own PDFs. Questions will search only your papers.";
  } else if (mode === "ask") {
    hint = scope === "mine"
      ? `Ask searches only your ${myDocs.length === 1 ? "paper" : `${myDocs.length} papers`}. Click a book to ask about it.`
      : `Ask searches these ${libraryDocs.length} papers. Click a book to ask about it.`;
  } else {
    hint = `${selected.length} of ${scopeDocs.length} picked. Click books to pick them, or pick all.`;
  }

  let caption;
  if (hovered) {
    caption = (
      <>
        <strong>{prettyName(hovered)}</strong>
        {mode === "ask" ? "  ·  click to ask about it" : selected.includes(hovered) ? "  ·  picked, click to put it back" : "  ·  click to pick it"}
      </>
    );
  } else if (cited.size > 0) {
    caption = `Glowing on the shelf: the ${cited.size === 1 ? "paper" : `${cited.size} papers`} this answer came from.`;
  } else {
    caption = "Hover over a book to see its title.";
  }

  let index = 0;

  return (
    <>
      <style>{css}</style>
      <div className="ambient" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />
      <header className="topbar">
        <div className="topbar-inner">
          <span className="brandmark"><i aria-hidden="true" /><span>Study Assistant AI</span></span>
          <nav className="nav" aria-label="Main">
            <a href="#ask" aria-current={view === "ask" ? "page" : undefined}>Ask</a>
            <a href="#compare" aria-current={view === "compare" ? "page" : undefined}>Compare</a>
            <a href="#about" aria-current={view === "about" ? "page" : undefined}>About</a>
          </nav>
          <a className="gh" href={REPO_URL} target="_blank" rel="noreferrer"><GitHubIcon /><span>Source</span></a>
        </div>
      </header>
      <div className={`page view-${view} ${mode === "compare" ? "compare-mode" : ""}`}>

        {view === "about" ? (
          <main>
            <section className="about" id="about" aria-labelledby="about-title">
              <div className="about-head">
                <p className="eyebrow">About</p>
                <h2 id="about-title">A research assistant that <em>shows its sources</em></h2>
                <p>
                  Study Assistant AI reads a library of research papers and answers questions from them, pointing to the exact
                  papers each answer came from. When the answer isn't in the papers, it says so instead of guessing. It was built
                  end to end, from PDF parsing to a fine-tuned local model, and every part is measured with its own benchmark.
                </p>
              </div>

              <h3>How it works</h3>
              <div className="how">
                <article className="panel how-card">
                  <span className="num">01 / Ask</span>
                  <h4>Answers with citations</h4>
                  <p>Keyword and meaning-based search find the best passages, a reranker sorts them, and the model answers only from those passages and names the ones it used.</p>
                  <div className="flow">BM25 + vectors → cross-encoder → gpt-4o-mini</div>
                </article>
                <article className="panel how-card">
                  <span className="num">02 / Compare</span>
                  <h4>Many papers, one table</h4>
                  <p>The relevant papers are picked from their abstracts, facts are pulled from each paper in parallel, and everything is merged into a single table with notes on what was left out.</p>
                  <div className="flow">select → extract → reduce</div>
                </article>
                <article className="panel how-card">
                  <span className="num">03 / Fine-tune</span>
                  <h4>A local model that refuses to guess</h4>
                  <p>A 3B Llama model was fine-tuned with QLoRA on a 6 GB laptop GPU to answer from retrieved passages, cite them, and refuse when the answer isn't there.</p>
                  <div className="flow">Llama-3.2-3B · QLoRA · Unsloth</div>
                </article>
              </div>

              <h3>Results</h3>
              <div className="results">
                <div className="panel metric"><b>30/30</b><span>Ask benchmark</span><small>right paper cited and right fact, plus 5/5 correct refusals</small></div>
                <div className="panel metric"><b>1.00</b><span>Compare precision</span><small>no wrong papers in the table, 0.92 recall</small></div>
                <div className="panel metric"><b>8%</b><span>Hallucination, local 3B</span><small>vs 26% for gpt-4o-mini on unseen papers</small></div>
                <div className="panel metric"><b>85%</b><span>Citation accuracy</span><small>vs 81% for gpt-4o-mini</small></div>
              </div>

              <h3>Built with</h3>
              <div className="stack">
                {["Python", "FastAPI", "React", "ChromaDB", "sentence-transformers", "BM25", "Cross-encoder rerank", "OpenAI gpt-4o / 4o-mini", "Unsloth", "QLoRA", "Llama-3.2-3B"].map((t) => (
                  <span key={t}>{t}</span>
                ))}
              </div>

              <h3>Built by</h3>
              <div className="panel builder">
                <div className="builder-who">
                  <div className="avatar" aria-hidden="true">PK</div>
                  <div>
                    <strong>Prerit Kumar</strong>
                    <p>Final-year Data Science student · building with LLMs, retrieval and fine-tuning</p>
                  </div>
                </div>
                <a className="gh" href={REPO_URL} target="_blank" rel="noreferrer"><GitHubIcon />View the code</a>
              </div>
            </section>
          </main>
        ) : (
          <>
            <div className="intro-head reveal" style={{ "--d": "60ms" }}>
              {view === "ask" ? (
                <>
                  <h1>Ask your <em>papers</em></h1>
                  <p>Every answer shows the papers it came from. If the answer isn't in them, it says so.</p>
                </>
              ) : (
                <>
                  <h1>Compare <em>papers</em></h1>
                  <p>Pick papers on the shelf and get one table that lines them up, with notes on what was left out.</p>
                </>
              )}
            </div>

            <section className="panel shelf-wrap reveal" style={{ "--d": "160ms" }} aria-label="Paper shelf">
              <div className="shelf-bar">
                <div>
                  <div className="scope" role="group" aria-label="Which papers">
                    <button aria-pressed={scope === "library"} onClick={() => switchScope("library")}>
                      Demo library <span>{libraryDocs.length}</span>
                    </button>
                    <button aria-pressed={scope === "mine"} onClick={() => switchScope("mine")}>
                      My papers <span>{myDocs.length}</span>
                    </button>
                  </div>
                  <p className="hint">{hint}</p>
                </div>
                <div className="shelf-tools">
                  {scope === "library" && (
                    <input
                      className="filter"
                      type="search"
                      placeholder="Find a paper"
                      value={filter}
                      onChange={(e) => setFilter(e.target.value)}
                      aria-label="Find a paper"
                    />
                  )}
                  {mode === "compare" && scopeDocs.length > 0 && (
                    <>
                      <button className="text-btn" onClick={() => setSelected(scopeDocs)}>Pick all</button>
                      <button className="text-btn" onClick={() => setSelected([])}>Clear</button>
                    </>
                  )}
                  {UPLOADS_ON && (
                    <button className="upload-btn" onClick={() => fileInput.current?.click()}>+ Upload PDF</button>
                  )}
                </div>
              </div>

              {scope === "mine" && myDocs.length === 0 ? (
                <button
                  className={`empty-shelf ${dragging ? "dragging" : ""}`}
                  onClick={() => fileInput.current?.click()}
                  disabled={!UPLOADS_ON}
                  {...dropHandlers}
                >
                  <strong>Your shelf is empty</strong>
                  {UPLOADS_ON ? "Drop a PDF here or click to upload. Ask and Compare will use only your papers." : "Uploads are turned off here."}
                </button>
              ) : (
                <div
                  className={`shelf ${intro ? "intro" : ""} ${loading ? "scanning" : ""} ${dragging ? "dragging" : ""}`}
                  {...dropHandlers}
                  onMouseMove={onShelfMove}
                  onMouseLeave={() => setTilt(null)}
                >
                  <div className="bookcase" style={tilt ? { "--rx": `${tilt.rx}deg`, "--ry": `${tilt.ry}deg` } : undefined}>
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
                          {scope === "mine" && UPLOADS_ON && (
                            <button className="add-spine" onClick={() => fileInput.current?.click()} aria-label="Upload a PDF" title="Upload a PDF">
                              +
                            </button>
                          )}
                        </div>
                        <span className="section-name" style={{ "--dot": section.color }}>{section.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <input
                ref={fileInput}
                type="file"
                accept=".pdf"
                hidden
                onChange={(e) => { uploadFile(e.target.files[0]); e.target.value = ""; }}
              />
              {!(scope === "mine" && myDocs.length === 0) && <p className="caption" aria-live="polite">{caption}</p>}
              {scope === "mine" && myDocs.length > 0 && UPLOADS_ON && (
                <div className="my-files">
                  <span className="label">Your files</span>
                  {myDocs.map((doc) => (
                    <span className="file-chip" key={doc} title={prettyName(doc)}>
                      <span>{prettyName(doc)}</span>
                      <button
                        className={confirmDelete === doc ? "confirm" : ""}
                        onClick={() => deleteFile(doc)}
                        disabled={deleting === doc}
                        aria-label={confirmDelete === doc ? `Confirm deleting ${prettyName(doc)}` : `Delete ${prettyName(doc)}`}
                      >
                        {deleting === doc ? "…" : confirmDelete === doc ? "Delete?" : "×"}
                      </button>
                    </span>
                  ))}
                </div>
              )}
              {uploadStatus.text && <p className={`status ${uploadStatus.kind}`}>{uploadStatus.text}</p>}
            </section>

            <main className="desk reveal" style={{ "--d": "260ms" }}>
              {mode === "ask" && (
                <div className="engine">
                  <span className="label">Answer with</span>
                  <div className="engine-opts" role="group" aria-label="Answer with">
                    <button aria-pressed={engine === "gpt"} onClick={() => setEngine("gpt")}>GPT-4o-mini</button>
                    <button
                      aria-pressed={engine === "both"}
                      onClick={() => setEngine("both")}
                      disabled={!localReady && engine !== "both"}
                      title={localReady ? "Ask GPT and the fine-tuned model, and compare" : "Start finetune/model_server.py to enable"}
                    >
                      GPT + my fine-tuned model
                    </button>
                  </div>
                  <span className="engine-note">
                    <span className={`dot ${localReady ? "on" : ""}`} aria-hidden="true" />
                    {localReady ? "Fine-tuned model running on the local GPU" : "Fine-tuned model not running"}
                  </span>
                </div>
              )}

              <div className="ask-row">
                <input
                  ref={questionInput}
                  className="question"
                  type="text"
                  value={question}
                  maxLength={MAX_QUESTION_LENGTH}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && run()}
                  placeholder={mode === "ask" ? "Ask anything about the papers…" : "What should the papers be compared on?"}
                  aria-label="Question"
                />
                <button className="primary" onClick={() => run()} disabled={loading}>
                  {loading ? "Reading…" : mode === "ask" ? "Ask" : "Compare"}
                </button>
              </div>

              {scopeDocs.length > 0 && (
                <div className="examples">
                  <span className="label">Try</span>
                  {examples.map((ex) => (
                    <button key={ex} className="example" onClick={() => pickExample(ex)} disabled={loading}>
                      {ex}
                    </button>
                  ))}
                </div>
              )}

              {loading && (
                <section className="panel result" aria-live="polite">
                  <div className="loading">
                    <p>
                      {mode === "ask" && engine === "both"
                        ? "Asking GPT and the fine-tuned model the same question…"
                        : mode === "ask"
                        ? `Pulling the most relevant passages from ${scopeDocs.length} papers…`
                        : `Reading the abstracts of ${selected.length} papers, then extracting facts…`}
                    </p>
                    <div className="bar" />
                    <div className="bar" />
                    <div className="bar" />
                  </div>
                </section>
              )}

              {!loading && result && (
                <section className="panel result fresh" aria-live="polite">
                  <div className="result-head">
                    <span className="label">{result.isError ? "Something went wrong" : isCompareResult ? "Comparison" : "Answer"}</span>
                    {!result.isError && (
                      <button className="copy" onClick={copyAnswer}>{copied ? "Copied" : "Copy"}</button>
                    )}
                  </div>
                  <div className={`result-grid ${isCompareResult ? "wide" : ""}`}>
                    <div className={`answer ${result.isError ? "error" : ""}`}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{mainText}</ReactMarkdown>
                    </div>
                    {result.sources.length > 0 && (
                      <aside className="margin">
                        <span className="label">{isCompareResult ? "Papers in the table" : "Sources"}</span>
                        <ol>
                          {result.sources.map((s, i) => (
                            <li key={s} style={{ "--i": i }}>{prettyName(s)}</li>
                          ))}
                        </ol>
                      </aside>
                    )}
                  </div>

                  {result.both && <ModelCheck both={result.both} />}

                  {notes && (
                    <details className="notes">
                      <summary>Why some papers aren't in the table</summary>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{notes}</ReactMarkdown>
                    </details>
                  )}
                </section>
              )}
            </main>
          </>
        )}

        <footer className="footer">
          <div>© 2026 Prerit Kumar</div>
          <a href={REPO_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
        </footer>
      </div>
    </>
  );
}

export default App;
