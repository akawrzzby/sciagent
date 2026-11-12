import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// The dev server proxies /api to the FastAPI backend so the frontend can be
// developed with hot reload against the real backend.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY ?? 'http://127.0.0.1:9000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1400,
    rollupOptions: {
      output: {
        manualChunks: {
          // Cytoscape and Mermaid are heavy and rarely change – split them out
          // so the main bundle stays small and the first paint is fast.
          cytoscape: ['cytoscape'],
          mermaid: ['mermaid'],
          markdown: ['react-markdown', 'remark-gfm', 'katex'],
        },
      },
    },
  },
})
