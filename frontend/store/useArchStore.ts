import { create } from "zustand"
import type {
  Architecture,
  ArchComponent,
  ChatMessage,
  BenchmarkResponse,
  UploadStatus,
  ResultsTab,
  CanvasPipeline,
} from "@/types/architecture"

interface ArchStore {
  // ── Session ─────────────────────────────────────────────
  sessionId: string | null
  setSessionId: (id: string) => void

  // ── Three-pipeline architectures ────────────────────────
  classical: Architecture | null
  hybrid: Architecture | null
  gemini: Architecture | null
  setPipelines: (
    classical: Architecture | null,
    hybrid: Architecture | null,
    gemini: Architecture | null,
  ) => void

  // ── Image ────────────────────────────────────────────────
  imageUrl: string | null
  setImageUrl: (url: string) => void
  // The name the user uploaded. image_url points at a UUID-named file, so it
  // cannot be used to identify which diagram this is.
  originalFilename: string | null
  setOriginalFilename: (name: string | null) => void

  // ── Upload State ─────────────────────────────────────────
  uploadStatus: UploadStatus
  setUploadStatus: (status: UploadStatus) => void
  uploadError: string | null
  setUploadError: (error: string | null) => void

  // ── Benchmark ────────────────────────────────────────────
  benchmark: BenchmarkResponse | null
  setBenchmark: (b: BenchmarkResponse | null) => void
  // Which session the displayed benchmark belongs to. Without this the panel
  // kept showing results from a previous upload after navigating to another
  // session, with nothing on screen to say they were stale.
  benchmarkSessionId: string | null
  setBenchmarkSessionId: (id: string | null) => void
  isBenchmarkLoading: boolean
  setIsBenchmarkLoading: (loading: boolean) => void

  // ── Canvas view ──────────────────────────────────────────
  canvasPipeline: CanvasPipeline
  setCanvasPipeline: (p: CanvasPipeline) => void
  showPackets: boolean
  toggleShowPackets: () => void
  hoveredNodeId: string | null
  setHoveredNodeId: (id: string | null) => void

  // ── Selected Component (for Explain tab) ────────────────
  selectedComponent: ArchComponent | null
  setSelectedComponent: (component: ArchComponent | null) => void

  // ── Right panel tabs ─────────────────────────────────────
  activeTab: ResultsTab
  setActiveTab: (tab: ResultsTab) => void

  // ── Chat ─────────────────────────────────────────────────
  chatMessages: ChatMessage[]
  addChatMessage: (message: ChatMessage) => void
  clearChat: () => void
  interviewMode: boolean
  toggleInterviewMode: () => void
  isChatLoading: boolean
  setIsChatLoading: (loading: boolean) => void

  // ── Reset ────────────────────────────────────────────────
  resetAll: () => void
}

export const useArchStore = create<ArchStore>((set) => ({
  // Session. Changing session invalidates any benchmark on screen — it was
  // scored against the previous session's extraction.
  sessionId: null,
  setSessionId: (id) =>
    set((s) =>
      s.sessionId === id
        ? { sessionId: id }
        : { sessionId: id, benchmark: null, benchmarkSessionId: null }
    ),

  // Pipelines
  classical: null,
  hybrid: null,
  gemini: null,
  setPipelines: (classical, hybrid, gemini) => set({ classical, hybrid, gemini }),

  // Image
  imageUrl: null,
  setImageUrl: (url) => set({ imageUrl: url }),
  originalFilename: null,
  setOriginalFilename: (name) => set({ originalFilename: name }),

  // Upload
  uploadStatus: "idle",
  setUploadStatus: (status) => set({ uploadStatus: status }),
  uploadError: null,
  setUploadError: (error) => set({ uploadError: error }),

  // Benchmark
  benchmark: null,
  setBenchmark: (b) => set({ benchmark: b }),
  benchmarkSessionId: null,
  setBenchmarkSessionId: (id) => set({ benchmarkSessionId: id }),
  isBenchmarkLoading: false,
  setIsBenchmarkLoading: (loading) => set({ isBenchmarkLoading: loading }),

  // Canvas
  canvasPipeline: "gemini",
  setCanvasPipeline: (p) => set({ canvasPipeline: p }),
  showPackets: true,
  toggleShowPackets: () => set((s) => ({ showPackets: !s.showPackets })),
  hoveredNodeId: null,
  setHoveredNodeId: (id) => set({ hoveredNodeId: id }),

  // Selected component
  selectedComponent: null,
  setSelectedComponent: (component) => set({ selectedComponent: component }),

  // Tabs
  activeTab: "comparison",
  setActiveTab: (tab) => set({ activeTab: tab }),

  // Chat
  chatMessages: [],
  addChatMessage: (message) =>
    set((state) => ({ chatMessages: [...state.chatMessages, message] })),
  clearChat: () => set({ chatMessages: [] }),
  interviewMode: false,
  toggleInterviewMode: () =>
    set((state) => ({ interviewMode: !state.interviewMode })),
  isChatLoading: false,
  setIsChatLoading: (loading) => set({ isChatLoading: loading }),

  // Reset everything for new upload
  resetAll: () =>
    set({
      sessionId: null,
      classical: null,
      hybrid: null,
      gemini: null,
      imageUrl: null,
      originalFilename: null,
      uploadStatus: "idle",
      uploadError: null,
      benchmark: null,
      benchmarkSessionId: null,
      isBenchmarkLoading: false,
      canvasPipeline: "gemini",
      showPackets: true,
      hoveredNodeId: null,
      selectedComponent: null,
      activeTab: "comparison",
      chatMessages: [],
      interviewMode: false,
    }),
}))
