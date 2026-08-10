"use client"
import {
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
  type EdgeProps,
} from "@xyflow/react"
import { useArchStore } from "@/store/useArchStore"

// Lower number = faster (seconds per cycle). Speed by connection label.
const SPEEDS: Record<string, number> = {
  grpc: 0.8,
  tcp: 0.8,
  sql: 1.2,
  rest: 1.5,
  http: 1.5,
  https: 1.5,
  websocket: 1.2,
  async: 2.5,
  event: 2.5,
  default: 1.8,
}

function speedFor(label: string): number {
  const k = (label || "").toLowerCase()
  for (const key of Object.keys(SPEEDS)) {
    if (key !== "default" && k.includes(key)) return SPEEDS[key]
  }
  return SPEEDS.default
}

interface PacketEdgeData {
  label?: string
  color?: string
  [key: string]: unknown
}

export function PacketEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  })

  const showPackets = useArchStore((s) => s.showPackets)
  const hoveredNodeId = useArchStore((s) => s.hoveredNodeId)

  const d = (data ?? {}) as PacketEdgeData
  const related = hoveredNodeId === null || hoveredNodeId === source || hoveredNodeId === target
  const dur = speedFor(d.label ?? "")
  const color = d.color ?? "#818cf8"

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: related ? "#475569" : "#1e293b",
          strokeWidth: hoveredNodeId !== null && related ? 2 : 1.5,
          opacity: related ? 1 : 0.25,
          transition: "opacity 0.2s, stroke 0.2s",
        }}
      />

      {/* Animated packet dot — animateMotion follows the path and is zoom/pan safe */}
      {showPackets && related && (
        <circle r={4} fill={color} opacity={0.95}>
          <animateMotion dur={`${dur}s`} repeatCount="indefinite" path={edgePath} />
        </circle>
      )}

      {/* Connection label */}
      {d.label && related && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: "none",
            }}
            className="px-1.5 py-0.5 rounded bg-[#0b1022] border border-white/10 text-[9px] text-slate-400"
          >
            {d.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

export const packetEdgeTypes = { packet: PacketEdge }
