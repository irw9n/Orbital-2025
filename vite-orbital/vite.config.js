import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'

const certPath = './localhost+1.pem'; 
const keyPath = './localhost+1-key.pem'; 

const hasCerts = fs.existsSync(certPath) && fs.existsSync(keyPath);

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, 
    port: 5173,
    ...(hasCerts ? {
      https: {
        key: fs.readFileSync(keyPath),
        cert: fs.readFileSync(certPath),
      },
    } : {}), 
  }
})