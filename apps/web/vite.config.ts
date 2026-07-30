import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: process.env.GEOARCHIVE_API_URL ? {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": process.env.GEOARCHIVE_API_URL,
      "/health": process.env.GEOARCHIVE_API_URL,
      "/docs": process.env.GEOARCHIVE_API_URL,
      "/openapi.json": process.env.GEOARCHIVE_API_URL
    }
  } : undefined
});
