import axios from "axios"
import type {
  DualAnalyzeResponse,
  SessionDetailResponse,
  SessionListItem,
  ChatResponse,
  BenchmarkResponse,
  DashboardData,
  GroundTruthItem,
} from "@/types/architecture"

// Base URL from environment variable. In development: http://localhost:8000
export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 min — hybrid SAM on CPU is slow (~140s), all three run in parallel
})

// Resolve a relative backend image URL (/uploads/x.png) to an absolute URL
export const resolveImageUrl = (url: string): string =>
  url.startsWith("http") ? url : `${API_BASE}${url}`

// ── Diagram Analysis (dual pipeline) ──────────────────────
export const analyzeImage = async (
  file: File,
  promptVariant: string = "chain_of_thought"
): Promise<DualAnalyzeResponse> => {
  const formData = new FormData()
  formData.append("file", file)
  formData.append("prompt_variant", promptVariant)

  const response = await api.post<DualAnalyzeResponse>("/api/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  })
  return response.data
}

// ── Sessions ──────────────────────────────────────────────
export const getSession = async (
  sessionId: string
): Promise<SessionDetailResponse> => {
  const response = await api.get<SessionDetailResponse>(
    `/api/sessions/${sessionId}`
  )
  return response.data
}

export const getSessions = async (): Promise<SessionListItem[]> => {
  const response = await api.get<SessionListItem[]>("/api/sessions")
  return response.data
}

// ── Chat ──────────────────────────────────────────────────
export const sendChatMessage = async (
  sessionId: string,
  message: string,
  interviewMode: boolean = false,
  componentId?: string | null,
  pipeline?: string | null
): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>("/api/chat", {
    session_id: sessionId,
    message,
    interview_mode: interviewMode,
    // Selected component and viewed pipeline, so "this component" resolves and
    // the answer discusses the extraction actually on screen.
    component_id: componentId ?? null,
    pipeline: pipeline ?? null,
  })
  return response.data
}

// ── Benchmark ─────────────────────────────────────────────
export const runBenchmark = async (
  sessionId: string,
  diagramId: string
): Promise<BenchmarkResponse> => {
  const response = await api.post<BenchmarkResponse>("/api/benchmark", {
    session_id: sessionId,
    diagram_id: diagramId,
  })
  return response.data
}

export const getGroundTruthList = async (): Promise<GroundTruthItem[]> => {
  const response = await api.get<GroundTruthItem[]>("/api/ground-truth")
  return response.data
}

// ── Dashboard ─────────────────────────────────────────────
export const getDashboard = async (): Promise<DashboardData> => {
  const response = await api.get<DashboardData>("/api/dashboard")
  return response.data
}

export default api
