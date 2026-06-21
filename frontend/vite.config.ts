import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api to the local Django server so the browser makes same-origin calls
// (no CORS setup needed). Frontend code uses relative URLs like /api/tutor/ask/.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
