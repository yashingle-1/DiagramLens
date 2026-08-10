"use client"
import { AlertTriangle } from "lucide-react"

interface HallucinationBadgeProps {
  name: string
  variant?: "hallucinated" | "missed"
}

/**
 * Small chip marking a component as hallucinated (Gemini invented it, not found
 * by OCR) or missed (present in ground truth but classical CV did not detect it).
 */
export default function HallucinationBadge({ name, variant = "hallucinated" }: HallucinationBadgeProps) {
  const isHallucinated = variant === "hallucinated"
  const tooltip = isHallucinated
    ? "Gemini invented this — not found in image by OCR"
    : "Present in the diagram but missed by Classical CV"

  return (
    <span
      title={tooltip}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] border ${
        isHallucinated
          ? "bg-red-500/15 border-red-500/30 text-red-300"
          : "bg-amber-500/15 border-amber-500/30 text-amber-300"
      }`}
    >
      <AlertTriangle className="w-2.5 h-2.5" />
      {name}
    </span>
  )
}
