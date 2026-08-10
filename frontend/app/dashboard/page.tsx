"use client"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, Cell,
} from "recharts"
import {
  ArrowLeft, Layers, Trophy, Zap, AlertTriangle, BarChart3, Gauge,
} from "lucide-react"
import { getDashboard } from "@/lib/api"
import type { DashboardData } from "@/types/architecture"

const BLUE = "#3b82f6"      // classical
const VIOLET = "#8b5cf6"    // hybrid
const EMERALD = "#10b981"   // gemini
const INDIGO = "#6366f1"    // metric: component F1
const CYAN = "#22d3ee"      // metric: connection F1

const PIPELINE_LABEL: Record<string, string> = {
  classical: "Classical",
  hybrid: "Hybrid",
  gemini: "Gemini",
}
const PIPELINE_FILL: Record<string, string> = {
  classical: BLUE,
  hybrid: VIOLET,
  gemini: EMERALD,
}

const tooltipStyle = {
  backgroundColor: "#0b1022",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: "8px",
  fontSize: "12px",
  color: "#fff",
}

function f1Pct(v: number | null | undefined): number {
  return typeof v === "number" ? Math.round(v * 100) : 0
}

function Card({ title, icon: Icon, children }: { title: string; icon: typeof BarChart3; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
      <div className="flex items-center gap-2 mb-4">
        <Icon className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-white">{title}</h3>
      </div>
      {children}
    </div>
  )
}

export default function DashboardPage() {
  const router = useRouter()
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDashboard()
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-[#03050f] flex items-center justify-center">
        <div className="w-10 h-10 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const empty = !data || data.total_runs === 0

  // ── Transform data for charts ─────────────────────────────
  const PIPELINE_ORDER = ["classical", "hybrid", "gemini"]

  const overallData = data
    ? PIPELINE_ORDER.filter((p) => data.overall[p]).map((p) => ({
        pipeline: PIPELINE_LABEL[p] ?? p,
        "Component F1": f1Pct(data.overall[p].avg_component_f1),
        "Connection F1": f1Pct(data.overall[p].avg_connection_f1),
      }))
    : []

  const byGroup = (group: Record<string, Record<string, { avg_component_f1: number | null }>> | undefined) =>
    group
      ? Object.entries(group).map(([key, pipelines]) => ({
          name: key,
          Classical: f1Pct(pipelines.classical?.avg_component_f1),
          Hybrid: f1Pct(pipelines.hybrid?.avg_component_f1),
          Gemini: f1Pct(pipelines.gemini?.avg_component_f1),
        }))
      : []

  const complexityData = byGroup(data?.by_complexity)
  const standardData = byGroup(data?.by_standard)

  const speedData = data
    ? PIPELINE_ORDER.filter((p) => data.speed_comparison[p]).map((p) => ({
        pipeline: PIPELINE_LABEL[p] ?? p,
        ms: data.speed_comparison[p].avg_response_time_ms ?? 0,
        fill: PIPELINE_FILL[p] ?? BLUE,
      }))
    : []

  return (
    <div className="min-h-screen bg-[#03050f] text-white">
      <div className="fixed inset-0 bg-[linear-gradient(rgba(99,102,241,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(99,102,241,0.03)_1px,transparent_1px)] bg-[size:48px_48px] pointer-events-none" />

      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between px-6 py-3 border-b border-white/5">
        <div className="flex items-center gap-3">
          <button onClick={() => router.push("/")} className="p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Layers className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-bold text-sm">DiagramLens · Benchmark Dashboard</span>
          </div>
        </div>
        <span className="text-xs text-slate-500">{data?.total_runs ?? 0} benchmark runs</span>
      </nav>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-8">
        {empty ? (
          <div className="text-center py-32">
            <BarChart3 className="w-12 h-12 text-slate-700 mx-auto mb-4" />
            <p className="text-slate-400 mb-1">No benchmark data yet</p>
            <p className="text-sm text-slate-600 mb-6">
              Upload diagrams and run benchmarks to populate this dashboard.
            </p>
            <button onClick={() => router.push("/")} className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-sm font-medium">
              Upload a diagram
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Header stat cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card title="Overall F1 by Pipeline" icon={Trophy}>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={overallData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="pipeline" stroke="#64748b" fontSize={12} />
                    <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} unit="%" />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Bar dataKey="Component F1" fill={INDIGO} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Connection F1" fill={CYAN} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card title="Speed Comparison (avg ms)" icon={Zap}>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={speedData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis type="number" stroke="#64748b" fontSize={12} unit="ms" />
                    <YAxis type="category" dataKey="pipeline" stroke="#64748b" fontSize={12} width={70} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                    <Bar dataKey="ms" radius={[0, 4, 4, 0]}>
                      {speedData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-[10px] text-slate-500 mt-2 text-center">
                  Lower is better — the classical pipeline runs with zero API latency.
                </p>
              </Card>
            </div>

            {/* By complexity + standard */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card title="Component F1 by Complexity" icon={Gauge}>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={complexityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                    <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} unit="%" />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Bar dataKey="Classical" fill={BLUE} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Hybrid" fill={VIOLET} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Gemini" fill={EMERALD} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card title="Component F1 by Diagram Standard" icon={BarChart3}>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={standardData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                    <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} unit="%" />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Bar dataKey="Classical" fill={BLUE} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Hybrid" fill={VIOLET} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Gemini" fill={EMERALD} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            {/* Hallucination table */}
            <Card title="Gemini Hallucination Rate by Diagram" icon={AlertTriangle}>
              {data && data.hallucination_table.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-500 border-b border-white/5">
                        <th className="text-left py-2 font-medium">Diagram</th>
                        <th className="text-left py-2 font-medium">Standard</th>
                        <th className="text-left py-2 font-medium">Complexity</th>
                        <th className="text-right py-2 font-medium">Rate</th>
                        <th className="text-left py-2 font-medium pl-4">Hallucinated</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.hallucination_table.map((row) => (
                        <tr key={row.diagram_id} className="border-b border-white/5">
                          <td className="py-2 text-slate-300">{row.diagram_id}</td>
                          <td className="py-2 text-slate-500 uppercase">{row.diagram_standard}</td>
                          <td className="py-2 text-slate-500">{row.complexity}</td>
                          <td className="py-2 text-right">
                            <span className={row.hallucination_rate > 0 ? "text-red-400 font-semibold" : "text-green-400"}>
                              {Math.round(row.hallucination_rate * 100)}%
                            </span>
                          </td>
                          <td className="py-2 pl-4">
                            <div className="flex flex-wrap gap-1">
                              {row.hallucinated.length === 0 ? (
                                <span className="text-slate-600">none</span>
                              ) : (
                                row.hallucinated.map((n) => (
                                  <span key={n} className="px-1.5 py-0.5 rounded bg-red-500/15 text-red-300 text-[10px]">
                                    {n}
                                  </span>
                                ))
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-xs text-slate-500 text-center py-6">No hallucination data yet.</p>
              )}
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
