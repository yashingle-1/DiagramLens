"use client"
import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import {
  ArrowLeft, Layers, BarChart3, MessageSquare, GitCompareArrows,
  Image as ImageIcon, Play, Pause, Cpu, Sparkles, Boxes, Columns2,
  LayoutDashboard, ChevronLeft, ChevronRight,
} from "lucide-react"
import { useArchStore } from "@/store/useArchStore"
import { getSession, resolveImageUrl } from "@/lib/api"
import DiagramCanvas from "@/components/canvas/DiagramCanvas"
import ChatPanel from "@/components/chat/ChatPanel"
import BenchmarkPanel from "@/components/BenchmarkPanel"
import ComparisonPanel from "@/components/ComparisonPanel"
import type { CanvasPipeline, ResultsTab } from "@/types/architecture"

export default function ResultsPage() {
  const params = useParams()
  const router = useRouter()
  const sessionId = params.id as string

  const {
    classical, hybrid, gemini, imageUrl, setPipelines, setImageUrl, setSessionId,
    setOriginalFilename,
    canvasPipeline, setCanvasPipeline, showPackets, toggleShowPackets,
    activeTab, setActiveTab, selectedComponent,
  } = useArchStore()

  const [loading, setLoading] = useState(false)
  const [imageOpen, setImageOpen] = useState(true)

  // Load session on refresh / shared link
  useEffect(() => {
    if ((!classical && !hybrid && !gemini) && sessionId) {
      setLoading(true)
      getSession(sessionId)
        .then((s) => {
          setSessionId(s.session_id)
          setPipelines(s.classical, s.hybrid, s.gemini)
          setImageUrl(s.image_url)
          setOriginalFilename(s.original_filename ?? null)
        })
        .catch(() => router.push("/"))
        .finally(() => setLoading(false))
    } else if (sessionId) {
      setSessionId(sessionId)
    }
  }, [sessionId])

  if (loading || (!classical && !hybrid && !gemini)) {
    return (
      <div className="min-h-screen bg-[#03050f] flex items-center justify-center">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-400 text-sm">Loading all pipelines…</p>
        </div>
      </div>
    )
  }

  const tabs: { id: ResultsTab; label: string; icon: typeof Layers }[] = [
    { id: "comparison", label: "Comparison", icon: GitCompareArrows },
    { id: "benchmark", label: "Benchmark", icon: BarChart3 },
    { id: "explain", label: "Explain", icon: MessageSquare },
  ]

  const pipelineToggles: { id: CanvasPipeline; label: string; icon: typeof Cpu; color: string }[] = [
    { id: "classical", label: "Classical", icon: Cpu, color: "#3b82f6" },
    { id: "hybrid", label: "Hybrid", icon: Boxes, color: "#8b5cf6" },
    { id: "gemini", label: "Gemini", icon: Sparkles, color: "#10b981" },
    { id: "all", label: "All", icon: Columns2, color: "#818cf8" },
  ]

  const showClassical = canvasPipeline === "classical" || canvasPipeline === "all"
  const showHybrid = canvasPipeline === "hybrid" || canvasPipeline === "all"
  const showGemini = canvasPipeline === "gemini" || canvasPipeline === "all"

  return (
    <div className="h-screen bg-[#03050f] flex flex-col overflow-hidden text-white">
      {/* Top nav */}
      <nav className="flex items-center justify-between px-4 py-2.5 border-b border-white/5 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/")}
            className="p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Layers className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-bold text-sm tracking-tight">DiagramLens</span>
          </div>
          <div className="h-4 w-px bg-white/10" />
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <span className="text-blue-400">{classical?.components.length ?? 0}</span> classical
            <span className="mx-0.5">·</span>
            <span className="text-violet-400">{hybrid?.components.length ?? 0}</span> hybrid
            <span className="mx-0.5">·</span>
            <span className="text-emerald-400">{gemini?.components.length ?? 0}</span> gemini
          </div>
        </div>

        <button
          onClick={() => router.push("/dashboard")}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-white/5 transition-all"
        >
          <LayoutDashboard className="w-3.5 h-3.5" />
          Dashboard
        </button>
      </nav>

      <div className="flex-1 flex overflow-hidden">
        {/* Left — uploaded image */}
        <div className={`flex-shrink-0 border-r border-white/5 bg-[#070a18] transition-all duration-300 ${imageOpen ? "w-64" : "w-10"}`}>
          {imageOpen ? (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between px-3 py-2.5 border-b border-white/5">
                <div className="flex items-center gap-1.5 text-xs text-slate-400">
                  <ImageIcon className="w-3.5 h-3.5" /> Source
                </div>
                <button onClick={() => setImageOpen(false)} className="text-slate-500 hover:text-white">
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="flex-1 overflow-auto p-3">
                {imageUrl && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={resolveImageUrl(imageUrl)}
                    alt="Uploaded architecture diagram"
                    className="w-full rounded-lg border border-white/10"
                  />
                )}
              </div>
            </div>
          ) : (
            <button
              onClick={() => setImageOpen(true)}
              className="w-full h-full flex items-center justify-center text-slate-500 hover:text-white"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Center — canvas */}
        <div className="flex-1 relative flex flex-col">
          {/* Canvas toolbar */}
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 p-1 rounded-xl bg-black/60 backdrop-blur-md border border-white/10">
            {pipelineToggles.map((p) => (
              <button
                key={p.id}
                onClick={() => setCanvasPipeline(p.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all ${
                  canvasPipeline === p.id ? "bg-white/10 text-white" : "text-slate-400 hover:text-white"
                }`}
              >
                <p.icon className="w-3.5 h-3.5" style={{ color: canvasPipeline === p.id ? p.color : undefined }} />
                {p.label}
              </button>
            ))}
            <div className="w-px h-5 bg-white/10 mx-0.5" />
            <button
              onClick={toggleShowPackets}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-all ${
                showPackets ? "bg-indigo-500/20 text-indigo-300" : "text-slate-400 hover:text-white"
              }`}
              title="Toggle packet flow animation"
            >
              {showPackets ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              Packets
            </button>
          </div>

          {/* Canvas area */}
          <div className="flex-1 flex">
            {showClassical && classical && (
              <div className="flex-1 relative border-r border-white/5">
                <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-blue-500/15 border border-blue-500/30">
                  <Cpu className="w-3 h-3 text-blue-400" />
                  <span className="text-[10px] font-semibold text-blue-300">CLASSICAL CV</span>
                </div>
                <DiagramCanvas architecture={classical} />
              </div>
            )}
            {showHybrid && hybrid && (
              <div className="flex-1 relative border-r border-white/5">
                <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-violet-500/15 border border-violet-500/30">
                  <Boxes className="w-3 h-3 text-violet-400" />
                  <span className="text-[10px] font-semibold text-violet-300">HYBRID ML</span>
                </div>
                <DiagramCanvas architecture={hybrid} />
              </div>
            )}
            {showHybrid && !hybrid && (
              <div className="flex-1 relative border-r border-white/5 flex items-center justify-center">
                <p className="text-xs text-slate-600">No hybrid result for this session</p>
              </div>
            )}
            {showGemini && gemini && (
              <div className="flex-1 relative">
                <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/15 border border-emerald-500/30">
                  <Sparkles className="w-3 h-3 text-emerald-400" />
                  <span className="text-[10px] font-semibold text-emerald-300">GEMINI</span>
                </div>
                <DiagramCanvas
                  architecture={gemini}
                  hallucinatedNames={gemini.hallucinated_components ?? []}
                />
              </div>
            )}
          </div>
        </div>

        {/* Right — tabbed panel */}
        <div className="w-[380px] flex-shrink-0 border-l border-white/5 bg-[#0b1022] flex flex-col">
          {/* Tab switcher */}
          <div className="flex items-center gap-1 p-2 border-b border-white/5">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 rounded-lg text-xs transition-all ${
                  activeTab === t.id
                    ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                    : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
                }`}
              >
                <t.icon className="w-3.5 h-3.5" />
                {t.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === "comparison" && <ComparisonPanel />}
            {activeTab === "benchmark" && <BenchmarkPanel />}
            {activeTab === "explain" && (
              <div className="h-full flex flex-col">
                {selectedComponent && (
                  <div className="px-4 py-2.5 border-b border-white/5 bg-white/[0.02]">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider">Selected</p>
                    <p className="text-sm font-semibold text-white">{selectedComponent.name}</p>
                  </div>
                )}
                <div className="flex-1 overflow-hidden">
                  <ChatPanel />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
