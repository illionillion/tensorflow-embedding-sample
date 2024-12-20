import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import { visualizer } from "rollup-plugin-visualizer";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    visualizer({
      open: true, // ビルド後に自動でブラウザで開く
      filename: "bundle-analysis.html", // 出力ファイル名
    }),
  ],
  base: process.env.GITHUB_PAGES ? "/tensorflow-embedding-sample/" : "./",
  build: {
    outDir: "docs",
  },
});
