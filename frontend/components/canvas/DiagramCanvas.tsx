"use client"
import { useEffect, useMemo } from "react"
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  type Node,
  type Edge,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import dagre from "dagre"
import { nodeTypes } from "./nodes/ArchNode"
import { packetEdgeTypes } from "../PacketAnimation"
import type { Architecture } from "@/types/architecture"

// Packet/minimap colour per component type
const TYPE_COLORS: Record<string, string> = {
  service: "#3b82f6",
  database: "#10b981",
  gateway: "#f97316",
  queue: "#8b5cf6",
  cache: "#ef4444",
  cdn: "#0ea5e9",
  load_balancer: "#06b6d4",
  client: "#94a3b8",
  storage: "#14b8a6",
  monitoring: "#6366f1",
  notification: "#ec4899",
  other: "#94a3b8",
}

function getLayoutedElements(nodes: Node[], edges: Edge[]) {
  const dagreGraph = new dagre.graphlib.Graph()
  dagreGraph.setDefaultEdgeLabel(() => ({}))
  dagreGraph.setGraph({ rankdir: "TB", nodesep: 80, ranksep: 110 })

  nodes.forEach((node) => dagreGraph.setNode(node.id, { width: 190, height: 84 }))
  edges.forEach((edge) => dagreGraph.setEdge(edge.source, edge.target))

  dagre.layout(dagreGraph)

  const layoutedNodes = nodes.map((node) => {
    const pos = dagreGraph.node(node.id)
    return { ...node, position: { x: pos.x - 95, y: pos.y - 42 } }
  })
  return { nodes: layoutedNodes, edges }
}

function architectureToFlow(architecture: Architecture, hallucinatedNames: Set<string>) {
  const typeById = new Map<string, string>()
  architecture.components.forEach((c) => typeById.set(c.id, c.type))

  const nodes: Node[] = architecture.components.map((comp) => ({
    id: comp.id,
    type: comp.type,
    position: { x: 0, y: 0 },
    data: {
      component: comp,
      hallucinated: hallucinatedNames.has(comp.name),
    },
  }))

  const edges: Edge[] = architecture.connections.map((conn) => {
    const srcType = typeById.get(conn.source) || "other"
    return {
      id: conn.id,
      source: conn.source,
      target: conn.target,
      type: "packet",
      data: {
        label: conn.label || conn.protocol || "",
        color: TYPE_COLORS[srcType] || TYPE_COLORS.other,
      },
    }
  })

  return getLayoutedElements(nodes, edges)
}

interface DiagramCanvasProps {
  architecture: Architecture
  hallucinatedNames?: string[]
}

export default function DiagramCanvas({ architecture, hallucinatedNames = [] }: DiagramCanvasProps) {
  const hallucinatedSet = useMemo(() => new Set(hallucinatedNames), [hallucinatedNames])

  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => architectureToFlow(architecture, hallucinatedSet),
    [architecture, hallucinatedSet]
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges)

  useEffect(() => {
    const { nodes: n, edges: e } = architectureToFlow(architecture, hallucinatedSet)
    setNodes(n)
    setEdges(e)
  }, [architecture, hallucinatedSet, setNodes, setEdges])

  return (
    <div className="w-full h-full bg-[#03050f]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        edgeTypes={packetEdgeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1e293b" />
        <Controls className="!bg-slate-900 !border-slate-700 !shadow-lg" showInteractive={false} />
        <MiniMap
          className="!bg-slate-900 !border-slate-700"
          nodeColor={(node) => TYPE_COLORS[node.type || "other"] || "#64748b"}
          maskColor="rgba(0,0,0,0.7)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  )
}
