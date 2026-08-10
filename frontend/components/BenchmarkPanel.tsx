"use client"
import { useEffect, useState } from "react"
import { Play, Loader2, Trophy, AlertTriangle, Target, Zap, ChevronDown } from "lucide-react"
import { useArchStore } from "@/store/useArchStore"
import { getGroundTruthList, runBenchmark } from "@/lib/api"
import type { GroundTruthItem, BenchmarkResult } from "@/types/architecture"

function pct(n: number | undefined | null): string {
  if (typeof n !== "number") return "—"
  return `${Math.round(n * 100)}%`
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-[10px] mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-semibold">{pct(value)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${(value || 0) * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function isResult(r: unknown): r is BenchmarkResult {
  return !!r && typeof r === "object" && "component_f1" in (r as object)
}

function ScoreCard({ result, accent, title }: { result: BenchmarkResult; accent: string; title: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: accent }} />
          <span className="text-xs font-semibold text-white">{title}</span>
        </div>
        <div className="text-right">
          <span className="text-xl font-bold" style={{ color: accent }}>{pct(result.component_f1)}</span>
          <span className="text-[9px] text-slate-500 uppercase tracking-wider ml-1">F1</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2.5">
        <ScoreBar label="Prec" value={result.component_precision} color={accent} />
        <ScoreBar label="Rec" value={result.component_recall} color={accent} />
        <ScoreBar label="Conn F1" value={result.connection_f1} color={accent} />
      </div>

      {typeof result.response_time_ms === "number" && (
        <div className="flex items-center gap-1.5 mt-3 pt-2.5 border-t border-white/5 text-[10px] text-slate-400">
          <Zap className="w-3 h-3" />
          {result.response_time_ms.toLocaleString()} ms
        </div>
      )}
    </div>
  )
}

const PIPELINE_META: { key: "classical" | "hybrid" | "gemini"; title: string; accent: string }[] = [
  { key: "classical", title: "Classical CV", accent: "#3b82f6" },
  { key: "hybrid", title: "Hybrid ML", accent: "#8b5cf6" },
  { key: "gemini", title: "Gemini", accent: "#10b981" },
]

/** Best ground-truth match for an uploaded filename.
 *  Scoring a diagram against the wrong annotation silently produces a
 *  near-zero F1 that looks like a pipeline failure, so the panel guesses
 *  rather than defaulting to whichever file sorts first.
 *
 *  Matches on the ORIGINAL upload name: stored files are named with a UUID,
 *  which carries no information about which diagram they hold. */
function matchGroundTruth(filename: string | null, list: GroundTruthItem[]): string | null {
  if (!filename || list.length === 0) return null
  const stem = decodeURIComponent(filename.split("/").pop() ?? "")
    .replace(/\.[^.]+$/, "")
    .toLowerCase()
  if (!stem) return null

  const exact = list.find((gt) => gt.diagram_id.toLowerCase() === stem)
  if (exact) return exact.diagram_id

  const tokens = stem.split(/[^a-z0-9]+/).filter((t) => t.length > 2)
  if (tokens.length === 0) return null

  let best: { id: string; score: number } | null = null
  for (const gt of list) {
    const id = gt.diagram_id.toLowerCase()
    const score = tokens.reduce((n, t) => (id.includes(t) ? n + 1 : n), 0)
    if (score > 0 && (!best || score > best.score)) best = { id: gt.diagram_id, score }
  }
  return best?.id ?? null
}

export default function BenchmarkPanel() {
  const {
    sessionId, originalFilename, benchmark, setBenchmark,
    benchmarkSessionId, setBenchmarkSessionId,
    isBenchmarkLoading, setIsBenchmarkLoading,
  } = useArchStore()
  const [groundTruths, setGroundTruths] = useState<GroundTruthItem[]>([])
  const [selected, setSelected] = useState<string>("")
  const [autoMatched, setAutoMatched] = useState(false)
  const [ranAgainst, setRanAgainst] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getGroundTruthList()
      .then(setGroundTruths)
      .catch(() => setError("Could not load ground truth list"))
  }, [])

  // Pick the ground truth matching the uploaded image, falling back to the
  // first entry only when nothing matches.
  useEffect(() => {
    if (groundTruths.length === 0) return
    const guess = matchGroundTruth(originalFilename, groundTruths)
    setAutoMatched(guess !== null)
    setSelected(guess ?? groundTruths[0].diagram_id)
  }, [originalFilename, groundTruths])

  // A benchmark belongs to the session it was scored against.
  const isStale = benchmark !== null && benchmarkSessionId !== sessionId

  const handleRun = async () => {
    if (!sessionId || !selected) return
    setError(null)
    setIsBenchmarkLoading(true)
    try {
      const result = await runBenchmark(sessionId, selected)
      setBenchmark(result)
      setBenchmarkSessionId(sessionId)
      setRanAgainst(selected)
    } catch (e) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string }
      setError(err?.response?.data?.detail || err?.message || "Benchmark failed")
    } finally {
      setIsBenchmarkLoading(false)
    }
  }

  // Resolve each pipeline's result (or null if errored / missing)
  const results = PIPELINE_META.map((m) => {
    const r = benchmark?.[m.key]
    return { ...m, result: isResult(r) ? r : null }
  })

  // Winner = highest component F1 among pipelines that produced a result
  const scored = results.filter((r) => r.result !== null)
  let winnerTitle: string | null = null
  if (scored.length > 0) {
    const best = scored.reduce((a, b) => (b.result!.component_f1 > a.result!.component_f1 ? b : a))
    winnerTitle = best.title
  }

  const geminiResult = results.find((r) => r.key === "gemini")?.result

  return (
    <div className="h-full flex flex-col">
      {/* Ground truth selector */}
      <div className="p-4 border-b border-white/5">
        <label className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 block">
          Ground truth diagram
        </label>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <select
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              className="w-full appearance-none bg-white/5 border border-white/10 rounded-lg pl-3 pr-8 py-2 text-xs text-white focus:outline-none focus:border-indigo-500/50"
            >
              {groundTruths.map((gt) => (
                <option key={gt.diagram_id} value={gt.diagram_id} className="bg-slate-900">
                  {gt.diagram_id} ({gt.diagram_standard}, {gt.complexity})
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
          </div>
          <button
            onClick={handleRun}
            disabled={!sessionId || !selected || isBenchmarkLoading}
            className="px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium text-white flex items-center gap-1.5 transition-colors"
          >
            {isBenchmarkLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            Score
          </button>
        </div>
        <p className="text-[10px] text-slate-600 mt-2">
          {autoMatched
            ? "Matched to your uploaded file automatically — change it if this is wrong."
            : "No ground truth matched this filename. Pick the right one before scoring."}
        </p>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-4">
        {error && (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300 mb-4">
            {error}
          </div>
        )}

        {/* Provenance. Scores are only meaningful for the session and ground
            truth they were computed against, so both are stated. */}
        {benchmark && (
          isStale ? (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300 mb-4">
              These scores were computed for a different session. Press Score to
              re-run them against the diagram currently open.
            </div>
          ) : (
            <p className="text-[10px] text-slate-500 mb-4">
              Scored against <span className="text-slate-300">{ranAgainst ?? selected}</span>
              {" · session "}
              <span className="text-slate-300">{sessionId?.slice(0, 8)}</span>
            </p>
          )
        )}

        {!benchmark && !error && (
          <div className="text-center mt-12">
            <Target className="w-10 h-10 text-slate-700 mx-auto mb-3" />
            <p className="text-sm text-slate-500">No benchmark run yet</p>
            <p className="text-xs text-slate-600 mt-1">
              Select a ground truth diagram and hit Score
            </p>
          </div>
        )}

        {benchmark && (
          <div className="space-y-3">
            {/* Score cards — one per pipeline */}
            {results.map((r) =>
              r.result ? (
                <ScoreCard key={r.key} result={r.result} accent={r.accent} title={r.title} />
              ) : (
                <div
                  key={r.key}
                  className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-xs text-slate-500 flex items-center gap-2"
                >
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: r.accent }} />
                  {r.title}: no result
                </div>
              )
            )}

            {/* Winner */}
            {winnerTitle && (
              <div className="flex items-center justify-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 py-2 text-xs">
                <Trophy className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-amber-300">
                  Highest component F1: <span className="font-semibold">{winnerTitle}</span>
                </span>
              </div>
            )}

            {/* Hallucinated by Gemini */}
            {geminiResult && geminiResult.hallucinated_components.length > 0 && (
              <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-3">
                <div className="flex items-center gap-1.5 mb-2">
                  <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
                  <span className="text-xs font-semibold text-red-300">
                    Hallucinated by Gemini ({geminiResult.hallucinated_components.length})
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {geminiResult.hallucinated_components.map((n) => (
                    <span key={n} className="px-2 py-0.5 rounded-full bg-red-500/15 border border-red-500/30 text-[10px] text-red-300">
                      {n}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Missed components — per pipeline */}
            {results.map((r) =>
              r.result && r.result.missed_components.length > 0 ? (
                <div key={`missed-${r.key}`} className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
                  <div className="flex items-center gap-1.5 mb-2">
                    <Target className="w-3.5 h-3.5 text-amber-400" />
                    <span className="text-xs font-semibold text-amber-300">
                      Missed by {r.title} ({r.result.missed_components.length})
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {r.result.missed_components.map((n) => (
                      <span key={n} className="px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-[10px] text-amber-300">
                        {n}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null
            )}
          </div>
        )}
      </div>
    </div>
  )
}
