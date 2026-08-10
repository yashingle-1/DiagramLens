"use client"
import { Handle, Position, NodeProps } from "@xyflow/react"
import {
  Database, Server, GitBranch, Mail, Zap, Globe, Scale,
  Monitor, HardDrive, Bell, AlertTriangle,
} from "lucide-react"
import { useArchStore } from "@/store/useArchStore"
import type { ArchComponent } from "@/types/architecture"

// Risk colour mapping
const riskColors: Record<string, string> = {
  low: "text-green-400",
  medium: "text-amber-400",
  high: "text-red-400",
}

// Node style per component type
type Style = { bg: string; border: string; glow: string; icon: typeof Server; label: string }
const nodeStyles: Record<string, Style> = {
  service:       { bg: "bg-blue-500/10",   border: "border-blue-500/40",   glow: "shadow-blue-500/20",   icon: Server,    label: "Service" },
  database:      { bg: "bg-green-500/10",  border: "border-green-500/40",  glow: "shadow-green-500/20",  icon: Database,  label: "Database" },
  gateway:       { bg: "bg-orange-500/10", border: "border-orange-500/40", glow: "shadow-orange-500/20", icon: GitBranch, label: "Gateway" },
  queue:         { bg: "bg-purple-500/10", border: "border-purple-500/40", glow: "shadow-purple-500/20", icon: Mail,      label: "Queue" },
  cache:         { bg: "bg-red-500/10",    border: "border-red-500/40",    glow: "shadow-red-500/20",    icon: Zap,       label: "Cache" },
  cdn:           { bg: "bg-sky-500/10",    border: "border-sky-500/40",    glow: "shadow-sky-500/20",    icon: Globe,     label: "CDN" },
  load_balancer: { bg: "bg-cyan-500/10",   border: "border-cyan-500/40",   glow: "shadow-cyan-500/20",   icon: Scale,     label: "Load Balancer" },
  client:        { bg: "bg-slate-500/10",  border: "border-slate-500/40",  glow: "shadow-slate-500/20",  icon: Monitor,   label: "Client" },
  storage:       { bg: "bg-teal-500/10",   border: "border-teal-500/40",   glow: "shadow-teal-500/20",   icon: HardDrive, label: "Storage" },
  monitoring:    { bg: "bg-indigo-500/10", border: "border-indigo-500/40", glow: "shadow-indigo-500/20", icon: Monitor,   label: "Monitoring" },
  notification:  { bg: "bg-pink-500/10",   border: "border-pink-500/40",   glow: "shadow-pink-500/20",   icon: Bell,      label: "Notification" },
  other:         { bg: "bg-slate-500/10",  border: "border-slate-500/40",  glow: "shadow-slate-500/20",  icon: Server,    label: "Other" },
}

interface NodeData {
  component: ArchComponent
  hallucinated?: boolean
  [key: string]: unknown
}

function BaseNode({ data, selected }: NodeProps) {
  const nodeData = data as NodeData
  const component = nodeData.component
  const hallucinated = nodeData.hallucinated === true
  const style = nodeStyles[component.type] || nodeStyles.other
  const Icon = style.icon
  const risk = component.metadata?.bottleneck_risk
  const confidence = component.confidence

  const { setSelectedComponent, setActiveTab, hoveredNodeId, setHoveredNodeId } = useArchStore()
  const isDimmed = hoveredNodeId !== null && hoveredNodeId !== component.id

  const handleClick = () => {
    setSelectedComponent(component)
    setActiveTab("explain")
  }

  return (
    <div
      onClick={handleClick}
      onMouseEnter={() => setHoveredNodeId(component.id)}
      onMouseLeave={() => setHoveredNodeId(null)}
      className={`
        group relative min-w-[150px] max-w-[190px] rounded-xl border px-3 py-2.5
        cursor-pointer transition-all duration-200 backdrop-blur-sm
        ${style.bg}
        ${hallucinated ? "border-red-500 border-dashed border-2" : style.border}
        ${selected ? "ring-2 ring-indigo-400 ring-offset-2 ring-offset-[#03050f]" : ""}
        ${isDimmed ? "opacity-35" : "opacity-100"}
        hover:scale-[1.04] hover:shadow-xl ${style.glow}
      `}
    >
      <Handle type="target" position={Position.Top} className="!bg-slate-500 !border-slate-400 !w-2 !h-2" />
      <Handle type="source" position={Position.Bottom} className="!bg-slate-500 !border-slate-400 !w-2 !h-2" />
      <Handle type="target" position={Position.Left} className="!bg-slate-500 !border-slate-400 !w-2 !h-2" />
      <Handle type="source" position={Position.Right} className="!bg-slate-500 !border-slate-400 !w-2 !h-2" />

      {/* Hallucination badge */}
      {hallucinated && (
        <div className="absolute -top-2.5 -right-2.5 z-10">
          <div
            className="flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-red-500 text-white text-[9px] font-bold shadow-lg"
            title="Gemini invented this — not found in image by OCR"
          >
            <AlertTriangle className="w-2.5 h-2.5" />
            HALLUCINATED
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 mb-1">
        <Icon className="w-3.5 h-3.5 text-slate-300 flex-shrink-0" />
        <span className="text-[10px] text-slate-400 uppercase tracking-wider">
          {style.label}
        </span>
      </div>

      <p className="text-xs font-semibold text-white leading-tight">
        {component.name}
      </p>

      {component.technology && (
        <p className="text-[10px] text-slate-400 mt-0.5">{component.technology}</p>
      )}

      <div className="flex items-center gap-2 mt-1.5">
        {risk && (
          <span className={`text-[9px] font-medium ${riskColors[risk] || "text-slate-400"}`}>
            ● {risk} risk
          </span>
        )}
        {typeof confidence === "number" && (
          <span className="text-[9px] text-purple-300 font-medium">
            {Math.round(confidence * 100)}% conf
          </span>
        )}
      </div>
    </div>
  )
}

export const nodeTypes = {
  service: BaseNode,
  database: BaseNode,
  gateway: BaseNode,
  queue: BaseNode,
  cache: BaseNode,
  cdn: BaseNode,
  load_balancer: BaseNode,
  client: BaseNode,
  storage: BaseNode,
  monitoring: BaseNode,
  notification: BaseNode,
  other: BaseNode,
}
