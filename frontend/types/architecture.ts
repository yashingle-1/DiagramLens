// ── Component Types ───────────────────────────────────────
export type ComponentType =
  | "service"
  | "database"
  | "gateway"
  | "queue"
  | "cache"
  | "cdn"
  | "load_balancer"
  | "client"
  | "storage"
  | "monitoring"
  | "notification"
  | "other"

export type RiskLevel = "low" | "medium" | "high"

export type Pipeline = "classical" | "hybrid" | "gemini"

export type DiagramStandard = "aws" | "c4" | "uml" | "informal"

export type Complexity = "low" | "medium" | "high"

export interface ComponentPosition {
  x: number
  y: number
}

export interface ComponentMetadata {
  role?: string
  bottleneck_risk?: RiskLevel
  scalability?: string
  security_surface?: RiskLevel
  responsibilities?: string[]
  suggestions?: string[]
}

export interface ArchComponent {
  id: string
  name: string
  type: ComponentType
  confidence?: number | null
  technology?: string | null
  position?: ComponentPosition | null
  metadata?: ComponentMetadata | null
}

// ── Connection Types ──────────────────────────────────────
export interface ArchConnection {
  id: string
  source: string
  target: string
  label?: string
  directed?: boolean
  direction?: "unidirectional" | "bidirectional" | string | null
  protocol?: string | null
  data_type?: string | null
}

// ── Full Architecture (one pipeline's output) ─────────────
export interface Architecture {
  session_id?: string
  pipeline?: Pipeline
  diagram_standard?: DiagramStandard
  complexity?: Complexity
  arch_type?: string
  components: ArchComponent[]
  connections: ArchConnection[]
  response_time_ms?: number
  confidence_score?: number | null
  hallucinated_components?: string[] | null
  hallucination_rate?: number | null
}

// ── API Response Types ────────────────────────────────────
export interface DualAnalyzeResponse {
  session_id: string
  classical: Architecture
  hybrid: Architecture | null
  gemini: Architecture
  image_url: string
  // Stored files are UUID-named, so this is the only way to tell which
  // diagram was uploaded.
  original_filename?: string | null
}

export interface SessionDetailResponse {
  session_id: string
  classical: Architecture | null
  hybrid: Architecture | null
  gemini: Architecture | null
  image_url: string
  created_at: string
  original_filename?: string | null
}

export interface SessionListItem {
  session_id: string
  original_filename?: string | null
  created_at: string
}

// ── Ground Truth ──────────────────────────────────────────
export interface GroundTruthItem {
  diagram_id: string
  diagram_standard: string
  complexity: string
  component_count: number
}

// ── Benchmark Types ───────────────────────────────────────
export interface BenchmarkResult {
  pipeline: Pipeline
  diagram_id: string
  diagram_standard: string
  complexity: string
  component_precision: number
  component_recall: number
  component_f1: number
  connection_precision: number
  connection_recall: number
  connection_f1: number
  hallucinated_components: string[]
  missed_components: string[]
  response_time_ms: number | null
  error?: string
}

export interface BenchmarkResponse {
  session_id: string
  diagram_id: string
  classical: BenchmarkResult | { error: string }
  hybrid: BenchmarkResult | { error: string }
  gemini: BenchmarkResult | { error: string }
}

// ── Dashboard Types ───────────────────────────────────────
export interface PipelineScore {
  avg_component_f1: number | null
  avg_connection_f1: number | null
  run_count?: number
}

export interface HallucinationRow {
  diagram_id: string
  diagram_standard: string
  complexity: string
  hallucinated: string[]
  hallucination_rate: number
  component_f1: number | null
}

export interface SpeedStat {
  avg_response_time_ms: number | null
  min_ms: number | null
  max_ms: number | null
}

export interface DashboardData {
  overall: Record<string, PipelineScore>
  by_complexity: Record<string, Record<string, PipelineScore>>
  by_standard: Record<string, Record<string, PipelineScore>>
  hallucination_table: HallucinationRow[]
  speed_comparison: Record<string, SpeedStat>
  total_runs: number
}

// ── Chat Types ────────────────────────────────────────────
export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  interview_mode: boolean
  created_at?: string
}

export interface ChatResponse {
  message: string
  session_id: string
  interview_mode: boolean
}

// ── UI State Types ────────────────────────────────────────
export type UploadStatus =
  | "idle"
  | "uploading"
  | "processing"
  | "done"
  | "error"

export type ResultsTab = "comparison" | "benchmark" | "explain"

export type CanvasPipeline = "classical" | "hybrid" | "gemini" | "all"
