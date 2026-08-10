"use client"
import { useMemo } from "react"
import { Cpu, Boxes, Sparkles, Zap, GitCompareArrows } from "lucide-react"
import { useArchStore } from "@/store/useArchStore"
import type { Architecture, ArchComponent } from "@/types/architecture"

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "")

// Lightweight fuzzy: shared if normalized names equal or one contains the other
function isShared(name: string, others: string[]): boolean {
  const n = norm(name)
  return others.some((o) => {
    const m = norm(o)
    return m === n || (m.length > 3 && n.length > 3 && (m.includes(n) || n.includes(m)))
  })
}

function PipelineHeader({
  icon: Icon, title, accent, arch,
}: {
  icon: typeof Cpu; title: string; accent: string; arch: Architecture | null
}) {
  return (
    <div className="flex items-center justify-between mb-2">
      <div className="flex items-center gap-1.5">
        <Icon className="w-3.5 h-3.5" style={{ color: accent }} />
        <span className="text-xs font-semibold text-white">{title}</span>
      </div>
      <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
        <span>{arch?.components.length ?? 0}c</span>
        <span>·</span>
        <span>{arch?.connections.length ?? 0}e</span>
        {typeof arch?.response_time_ms === "number" && (
          <>
            <span>·</span>
            <span className="flex items-center gap-0.5">
              <Zap className="w-2.5 h-2.5" />{arch.response_time_ms.toLocaleString()}ms
            </span>
          </>
        )}
      </div>
    </div>
  )
}

function ComponentRow({
  comp, accent, hallucinated, shared,
}: {
  comp: ArchComponent; accent: string; hallucinated: boolean; shared: boolean
}) {
  return (
    <div
      className={`flex items-center justify-between px-2 py-1.5 rounded-lg border text-[11px] ${
        hallucinated
          ? "border-red-500/40 border-dashed bg-red-500/5"
          : shared
            ? "border-white/5 bg-white/[0.02]"
            : "border-amber-500/20 bg-amber-500/5"
      }`}
    >
      <span className="text-slate-200 truncate">{comp.name}</span>
      <div className="flex items-center gap-1 flex-shrink-0">
        <span className="text-[8px] uppercase tracking-wide" style={{ color: accent }}>
          {comp.type.replace("_", " ")}
        </span>
        {hallucinated && (
          <span className="text-[8px] font-bold text-red-400" title="Not found in image by OCR">⚠</span>
        )}
      </div>
    </div>
  )
}

function PipelineColumn({
  icon, title, accent, arch, otherNames, hallucinatedSet,
}: {
  icon: typeof Cpu
  title: string
  accent: string
  arch: Architecture | null
  otherNames: string[]
  hallucinatedSet?: Set<string>
}) {
  return (
    <div>
      <PipelineHeader icon={icon} title={title} accent={accent} arch={arch} />
      <div className="space-y-1.5">
        {arch?.components.map((c) => (
          <ComponentRow
            key={c.id}
            comp={c}
            accent={accent}
            hallucinated={hallucinatedSet?.has(c.name) ?? false}
            shared={isShared(c.name, otherNames)}
          />
        ))}
        {(!arch || arch.components.length === 0) && (
          <p className="text-[10px] text-slate-600 px-2 py-1.5">No components</p>
        )}
      </div>
    </div>
  )
}

export default function ComparisonPanel() {
  const { classical, hybrid, gemini } = useArchStore()

  const hallucinatedSet = useMemo(
    () => new Set(gemini?.hallucinated_components ?? []),
    [gemini]
  )

  const classicalNames = useMemo(() => classical?.components.map((c) => c.name) ?? [], [classical])
  const hybridNames = useMemo(() => hybrid?.components.map((c) => c.name) ?? [], [hybrid])
  const geminiNames = useMemo(() => gemini?.components.map((c) => c.name) ?? [], [gemini])

  // A component is "shared" if it appears in any other pipeline.
  const classicalOthers = useMemo(() => [...hybridNames, ...geminiNames], [hybridNames, geminiNames])
  const hybridOthers = useMemo(() => [...classicalNames, ...geminiNames], [classicalNames, geminiNames])
  const geminiOthers = useMemo(() => [...classicalNames, ...hybridNames], [classicalNames, hybridNames])

  // Components found by all three (consensus) — measured against gemini as the reference set.
  const consensus = useMemo(
    () => geminiNames.filter((n) => isShared(n, classicalNames) && isShared(n, hybridNames)).length,
    [geminiNames, classicalNames, hybridNames]
  )

  const speedup = useMemo(() => {
    const c = classical?.response_time_ms
    const g = gemini?.response_time_ms
    if (c && g && c > 0) return (g / c).toFixed(1)
    return null
  }, [classical, gemini])

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* Diff summary */}
      <div className="grid grid-cols-4 gap-2">
        <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-2 text-center">
          <div className="text-lg font-bold text-blue-400">{classical?.components.length ?? 0}</div>
          <div className="text-[8px] text-slate-500 uppercase tracking-wide">Classical</div>
        </div>
        <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-2 text-center">
          <div className="text-lg font-bold text-violet-400">{hybrid?.components.length ?? 0}</div>
          <div className="text-[8px] text-slate-500 uppercase tracking-wide">Hybrid</div>
        </div>
        <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-2 text-center">
          <div className="text-lg font-bold text-emerald-400">{gemini?.components.length ?? 0}</div>
          <div className="text-[8px] text-slate-500 uppercase tracking-wide">Gemini</div>
        </div>
        <div className="rounded-lg border border-white/10 bg-white/[0.03] p-2 text-center">
          <div className="text-lg font-bold text-slate-300">{consensus}</div>
          <div className="text-[8px] text-slate-500 uppercase tracking-wide">All 3</div>
        </div>
      </div>

      {speedup && (
        <div className="flex items-center justify-center gap-2 rounded-lg border border-blue-500/20 bg-blue-500/5 py-2 text-xs">
          <GitCompareArrows className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-slate-300">
            Classical CV ran <span className="font-bold text-blue-400">{speedup}×</span> faster than Gemini
          </span>
        </div>
      )}

      {/* Three columns */}
      <div className="grid grid-cols-3 gap-2.5">
        <PipelineColumn
          icon={Cpu} title="Classical" accent="#3b82f6"
          arch={classical} otherNames={classicalOthers}
        />
        <PipelineColumn
          icon={Boxes} title="Hybrid" accent="#8b5cf6"
          arch={hybrid} otherNames={hybridOthers}
        />
        <PipelineColumn
          icon={Sparkles} title="Gemini" accent="#10b981"
          arch={gemini} otherNames={geminiOthers} hallucinatedSet={hallucinatedSet}
        />
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-white/5 text-[10px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-white/20" /> shared
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-amber-500/60" /> unique to pipeline
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full border border-dashed border-red-500" /> hallucinated
        </span>
      </div>
    </div>
  )
}
