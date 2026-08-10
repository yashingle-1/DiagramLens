"use client"
import { useRouter } from "next/navigation"
import { useCallback, useState, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import {
  Upload, Layers, Cpu, Boxes, Sparkles, BarChart3, ArrowRight, ChevronRight,
  LayoutDashboard, ScanSearch,
} from "lucide-react"
import { analyzeImage, getSessions } from "@/lib/api"
import { useArchStore } from "@/store/useArchStore"
import type { SessionListItem } from "@/types/architecture"

const PROMPT_VARIANTS = [
  { id: "chain_of_thought", label: "Chain of Thought" },
  { id: "few_shot", label: "Few Shot" },
  { id: "zero_shot", label: "Zero Shot" },
]

export default function HomePage() {
  const router = useRouter()
  const { setSessionId, setPipelines, setImageUrl, setOriginalFilename, setUploadStatus, setUploadError, resetAll } = useArchStore()
  const [isLoading, setIsLoading] = useState(false)
  const [promptVariant, setPromptVariant] = useState("chain_of_thought")
  const [sessions, setSessions] = useState<SessionListItem[]>([])

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    resetAll()
    setIsLoading(true)
    setUploadStatus("processing")

    try {
      const result = await analyzeImage(file, promptVariant)
      setSessionId(result.session_id)
      setPipelines(result.classical, result.hybrid, result.gemini)
      setImageUrl(result.image_url)
      setOriginalFilename(result.original_filename ?? file.name)
      setUploadStatus("done")
      router.push(`/results/${result.session_id}`)
    } catch (error) {
      const err = error as { response?: { data?: { detail?: string } } }
      setUploadStatus("error")
      setUploadError(err?.response?.data?.detail || "Analysis failed. Please try again.")
      setIsLoading(false)
    }
  }, [promptVariant])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxFiles: 1,
    disabled: isLoading,
  })

  useEffect(() => {
    getSessions().then(setSessions).catch(() => { })
  }, [])

  return (
    <main className="min-h-screen bg-[#03050f] text-white">
      <div className="fixed inset-0 bg-[linear-gradient(rgba(99,102,241,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(99,102,241,0.03)_1px,transparent_1px)] bg-[size:44px_44px] pointer-events-none" />

      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-5 border-b border-white/5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
            <Layers className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight">DiagramLens</span>
        </div>
        <button
          onClick={() => router.push("/dashboard")}
          className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <LayoutDashboard className="w-4 h-4" />
          Dashboard
        </button>
      </nav>

      <div className="relative z-10 max-w-4xl mx-auto px-8 pt-20 pb-16">
        {/* Hero */}
        <div className="text-center mb-14">
          <div className="inline-block px-3 py-1 text-xs tracking-widest uppercase text-indigo-400 border border-indigo-500/30 bg-indigo-500/10 rounded mb-6">
            Research
          </div>
          <h1 className="text-5xl font-bold leading-tight mb-6 bg-gradient-to-br from-white via-white to-slate-400 bg-clip-text text-transparent">
            Rule-Based CV, Specialized ML, or a VLM<br />Which Reads Architecture Diagrams Best?
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Upload an architecture diagram. Three pipelines extract it in parallel:
            a pure OpenCV + Tesseract baseline, a SAM + CLIP + TrOCR hybrid, and Gemini 2.5 Flash,
            then we score all three against ground truth.
          </p>
        </div>

        {/* Prompt variant selector */}
        <div className="flex justify-center gap-2 mb-6">
          {PROMPT_VARIANTS.map((v) => (
            <button
              key={v.id}
              onClick={() => setPromptVariant(v.id)}
              className={`px-3 py-1.5 text-xs rounded border transition-all ${promptVariant === v.id
                ? "border-indigo-500 bg-indigo-500/20 text-indigo-300"
                : "border-white/10 text-slate-500 hover:border-white/20"
                }`}
            >
              {v.label}
            </button>
          ))}
        </div>

        {/* Upload zone */}
        <div
          {...getRootProps()}
          className={`relative border-2 border-dashed rounded-xl p-16 text-center cursor-pointer transition-all duration-300 ${isDragActive
            ? "border-indigo-500 bg-indigo-500/10"
            : isLoading
              ? "border-white/10 bg-white/5 cursor-not-allowed"
              : "border-white/10 bg-white/5 hover:border-indigo-500/50 hover:bg-indigo-500/5"
            }`}
        >
          <input {...getInputProps()} />
          {isLoading ? (
            <div className="flex flex-col items-center gap-4">
              <div className="flex items-center gap-3">
                <Cpu className="w-6 h-6 text-blue-400 animate-pulse" />
                <span className="text-slate-500">+</span>
                <Boxes className="w-6 h-6 text-violet-400 animate-pulse" />
                <span className="text-slate-500">+</span>
                <Sparkles className="w-6 h-6 text-emerald-400 animate-pulse" />
              </div>
              <p className="text-slate-400">Running all three pipelines in parallel…</p>
              <p className="text-xs text-slate-600">Classical CV is instant · Hybrid ML ~1–2min · Gemini 5–15s</p>
            </div>
          ) : isDragActive ? (
            <div className="flex flex-col items-center gap-3">
              <Upload className="w-12 h-12 text-indigo-400" />
              <p className="text-indigo-300 font-medium">Drop your diagram here</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                <ScanSearch className="w-8 h-8 text-indigo-400" />
              </div>
              <div>
                <p className="text-white font-medium text-lg mb-1">Drop your architecture diagram here</p>
                <p className="text-slate-500 text-sm">PNG, JPG, WebP - AWS, C4, UML, or hand-drawn</p>
              </div>
              <div className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition-colors flex items-center gap-2">
                Browse Files <ArrowRight className="w-4 h-4" />
              </div>
            </div>
          )}
        </div>

        {/* Features */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-12">
          {[
            { icon: Cpu, color: "#3b82f6", title: "Classical CV", desc: "OpenCV contour detection + Tesseract OCR + Hough transform. Zero AI, zero API calls." },
            { icon: Boxes, color: "#8b5cf6", title: "Hybrid ML", desc: "SAM segmentation + CLIP classification + TrOCR text. Specialized models, no LLM, runs local." },
            { icon: Sparkles, color: "#10b981", title: "Gemini 2.5 Flash", desc: "A vision language model extracts the same schema with an OCR hallucination filter." },
            { icon: BarChart3, color: "#818cf8", title: "Benchmark & Score", desc: "Fuzzy-matched precision, recall, F1 and hallucination rate against ground truth." },
          ].map(({ icon: Icon, color, title, desc }) => (
            <div key={title} className="p-5 rounded-xl border border-white/5 bg-white/5">
              <Icon className="w-5 h-5 mb-3" style={{ color }} />
              <h3 className="font-semibold text-sm mb-2">{title}</h3>
              <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>

        {/* Recent sessions */}
        {sessions.length > 0 && (
          <div className="mt-12">
            <p className="text-xs text-slate-500 uppercase tracking-widest mb-4">Recent diagrams</p>
            <div className="space-y-2">
              {sessions.map((s) => (
                <div
                  key={s.session_id}
                  onClick={() => router.push(`/results/${s.session_id}`)}
                  className="flex items-center justify-between p-3 rounded-lg border border-white/5 bg-white/5 hover:border-indigo-500/30 cursor-pointer transition-all"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                      <Layers className="w-3.5 h-3.5 text-indigo-400" />
                    </div>
                    <div>
                      <p className="text-sm text-white">{s.original_filename || "Architecture diagram"}</p>
                      <p className="text-xs text-slate-500">{s.created_at?.slice(0, 10)}</p>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-600" />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
