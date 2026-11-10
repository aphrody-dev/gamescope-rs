// Zero-dependency Bun WebGPU Host Server for Gamescope-rs WebAssembly Stack
import { serve } from "bun";
import { join } from "path";
import { existsSync, readFileSync } from "fs";

const PORT = Number(process.env.PORT) || 8080;
const WEB_DIR = import.meta.dir;
const TARGET_WASM_DIR = join(WEB_DIR, "..", "target", "wasm32-unknown-unknown", "release");

console.log(`[gamescope-web] Initializing sovereign WebGPU server on http://localhost:${PORT}`);

serve({
  port: PORT,
  fetch(req) {
    const url = new URL(req.url);
    let pathname = url.pathname;

    if (pathname === "/" || pathname === "/index.html") {
      const htmlPath = join(WEB_DIR, "index.html");
      if (existsSync(htmlPath)) {
        return new Response(readFileSync(htmlPath), {
          headers: {
            "Content-Type": "text/html; charset=utf-8",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
          },
        });
      }
    }

    if (pathname === "/gamescope_web_wasm.wasm") {
      const wasmPath = join(TARGET_WASM_DIR, "gamescope_web_wasm.wasm");
      if (existsSync(wasmPath)) {
        return new Response(readFileSync(wasmPath), {
          headers: {
            "Content-Type": "application/wasm",
            "Cache-Control": "no-cache",
          },
        });
      }
    }

    // Serve local assets
    const localFilePath = join(WEB_DIR, pathname);
    if (existsSync(localFilePath) && !pathname.endsWith("/")) {
      const ext = pathname.split(".").pop();
      let contentType = "text/plain";
      if (ext === "html") contentType = "text/html";
      else if (ext === "js" || ext === "mjs") contentType = "application/javascript";
      else if (ext === "wasm") contentType = "application/wasm";
      else if (ext === "css") contentType = "text/css";
      else if (ext === "json") contentType = "application/json";

      return new Response(readFileSync(localFilePath), {
        headers: { "Content-Type": contentType },
      });
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`[gamescope-web] Serving WebGPU canvas at http://localhost:${PORT}`);
