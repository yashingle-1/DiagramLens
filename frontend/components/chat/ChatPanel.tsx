"use client"
import { useState, useRef, useEffect } from "react"
import { Send, Loader2, GraduationCap, Bot, User } from "lucide-react"
import { useArchStore } from "@/store/useArchStore"
import { sendChatMessage } from "@/lib/api"
import { v4 as uuidv4 } from "uuid"

// Quick question chips — reduces friction for new users
const QUICK_QUESTIONS = [
  "Is this scalable?",
  "Find bottlenecks",
  "10x traffic impact",
  "Security risks",
  "Explain like interview",
  "Compare to Netflix",
]

export default function ChatPanel() {
  const [input, setInput] = useState("")
  const bottomRef = useRef<HTMLDivElement>(null)

  const {
    sessionId,
    chatMessages,
    addChatMessage,
    isChatLoading,
    setIsChatLoading,
    interviewMode,
    toggleInterviewMode,
    selectedComponent,
    canvasPipeline,
  } = useArchStore()

  // Auto scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatMessages])

  const handleSend = async (message?: string) => {
    console.log("Session ID:", sessionId)
    const text = message || input.trim()
    if (!text || !sessionId || isChatLoading) return

    setInput("")

    addChatMessage({
      id: uuidv4(),
      role: "user",
      content: text,
      interview_mode: interviewMode,
    })

    setIsChatLoading(true)

    try {
      // The canvas can show "all"; that is not a stored pipeline, so let the
      // backend fall back to its default rather than querying for it.
      const pipeline = canvasPipeline === "all" ? null : canvasPipeline
      const response = await sendChatMessage(
        sessionId, text, interviewMode, selectedComponent?.id ?? null, pipeline
      )

      addChatMessage({
        id: uuidv4(),
        role: "assistant",
        content: response.message,
        interview_mode: interviewMode,
      })
    } catch (error: any) {
      console.error("Chat error:", error)
      addChatMessage({
        id: uuidv4(),
        role: "assistant",
        content: `Error: ${error?.response?.data?.detail || error?.message || "Unknown error"}`,
        interview_mode: false,
      })
    } finally {
      setIsChatLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="h-full flex flex-col bg-[#0b1022] border-l border-white/5">

      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/5">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-indigo-400" />
          <span className="font-semibold text-sm">AI Architect</span>
        </div>

        {/* Interview mode toggle */}
        <button
          onClick={toggleInterviewMode}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs border transition-all ${interviewMode
            ? "bg-indigo-500/20 border-indigo-500/40 text-indigo-300"
            : "bg-white/5 border-white/10 text-slate-500 hover:text-slate-300"
            }`}
        >
          <GraduationCap className="w-3 h-3" />
          Interview Mode
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chatMessages.length === 0 ? (
          <div className="text-center mt-8">
            <Bot className="w-10 h-10 text-slate-700 mx-auto mb-3" />
            <p className="text-sm text-slate-500 mb-1">
              Ask me anything about your architecture
            </p>
            <p className="text-xs text-slate-600">
              I have full context of every component and connection
            </p>
          </div>
        ) : (
          chatMessages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-2.5 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "assistant" && (
                <div className="w-6 h-6 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot className="w-3 h-3 text-indigo-400" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed ${msg.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-white/5 border border-white/5 text-slate-300"
                  }`}
              >
                {msg.content}
              </div>

              {msg.role === "user" && (
                <div className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User className="w-3 h-3 text-slate-400" />
                </div>
              )}
            </div>
          ))
        )}

        {/* Loading indicator */}
        {isChatLoading && (
          <div className="flex gap-2.5">
            <div className="w-6 h-6 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center flex-shrink-0">
              <Bot className="w-3 h-3 text-indigo-400" />
            </div>
            <div className="bg-white/5 border border-white/5 rounded-xl px-3.5 py-2.5">
              <Loader2 className="w-4 h-4 text-slate-400 animate-spin" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Quick questions */}
      {chatMessages.length === 0 && sessionId && (
        <div className="px-4 pb-2 flex flex-wrap gap-1.5">
          {QUICK_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => handleSend(q)}
              className="px-2.5 py-1 text-xs rounded-full border border-white/10 text-slate-400 hover:border-indigo-500/40 hover:text-indigo-300 transition-all"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-white/5">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              sessionId
                ? "Ask about your architecture..."
                : "Upload a diagram first"
            }
            disabled={!sessionId || isChatLoading}
            rows={1}
            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 resize-none focus:outline-none focus:border-indigo-500/50 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || !sessionId || isChatLoading}
            className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isChatLoading
              ? <Loader2 className="w-4 h-4 text-white animate-spin" />
              : <Send className="w-4 h-4 text-white" />
            }
          </button>
        </div>
        <p className="text-[10px] text-slate-600 mt-1.5 ml-1">
          Enter to send · Shift+Enter for new line
          {interviewMode && " · Interview mode ON"}
        </p>
      </div>
    </div>
  )
}
