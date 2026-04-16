import { defineConfig } from "vite";
import viteTsConfigPaths from "vite-tsconfig-paths";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    viteTsConfigPaths({ projects: ["./tsconfig.json"] }),
  ],
  server: {
    proxy: {
      "/api": "http://localhost:3000",
    },
  },
});
