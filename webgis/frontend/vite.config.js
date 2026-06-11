import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  publicDir: "dummy-data",
  server: {
    port: 5173,
    strictPort: false,
  },
});
