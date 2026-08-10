// "use client"
// import type { Metadata } from "next"
// import { Inter } from "next/font/google"
// import "./globals.css"
// import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
// import { useState } from "react"

// const inter = Inter({ subsets: ["latin"] })

// export default function RootLayout({
//   children,
// }: {
//   children: React.ReactNode
// }) {
//   // QueryClient created inside component so each session
//   // gets its own client — prevents state leaking between users
//   const [queryClient] = useState(() => new QueryClient({
//     defaultOptions: {
//       queries: {
//         staleTime: 1000 * 60 * 5,  // 5 minutes
//         retry: 1,
//       },
//     },
//   }))

//   return (
//     <html lang="en" className="dark">
//       <body className={inter.className}>
//         <QueryClientProvider client={queryClient}>
//           {children}
//         </QueryClientProvider>
//       </body>
//     </html>
//   )
// }


import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import Providers from "./providers"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "DiagramLens — Classical CV vs Hybrid ML vs Gemini",
  description: "Benchmark three paradigms — rule-based CV, a SAM+CLIP+TrOCR hybrid, and Gemini 2.5 Flash — on architecture diagram extraction",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}