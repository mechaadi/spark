import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import arraybuffer from "vite-plugin-arraybuffer";
import dts from "vite-plugin-dts";
import glsl from "vite-plugin-glsl";

const sparkRsDirectory = "rust/spark-rs/pkg";
if (!fs.existsSync(sparkRsDirectory)) {
  console.error(
    "\x1b[31m************************************************************************\x1b[0m",
  );
  console.error(
    "\x1b[31m Rust Wasm component not found, make sure to build them first.\x1b[0m",
  );
  console.error(
    "\x1b[31m Install Rust and run:\x1b[1m npm run build:wasm\x1b[0m",
  );
  console.error(
    "\x1b[31m************************************************************************\x1b[0m",
  );
  process.exit(1);
}

const assetsDirectory = "examples/assets";
const localAssetsDirectoryExist = fs.existsSync(assetsDirectory);
if (!localAssetsDirectoryExist) {
  console.log(
    "************************************************************************",
  );
  console.log(" Examples assets will be fetched from an external server.");
  console.log(
    " To work offline you can download them: npm run assets:download",
  );
  console.log(
    "************************************************************************",
  );
}

export default defineConfig(({ mode }) => {
  const isMinify = mode === "production";
  const isFirstPass = mode === "production";

  return {
    appType: "mpa",

    plugins: [
      arraybuffer(),
      glsl({
        include: ["**/*.glsl"],
      }),

      dts({ outDir: "dist/types" }),

      {
        name: "serve-node-modules-alias",
        configureServer(server) {
          const baseUrlPath = "/examples/js/vendor/";

          server.middlewares.use((req, res, next) => {
            if (!req.url.startsWith(baseUrlPath)) return next();

            const relModulePath = req.url.slice(baseUrlPath.length); // safe substring
            const absPath = path.resolve("node_modules", relModulePath);

            if (fs.existsSync(absPath) && fs.statSync(absPath).isFile()) {
              const ext = path.extname(absPath);
              const contentType =
                {
                  ".js": "application/javascript",
                  ".mjs": "application/javascript",
                  ".css": "text/css",
                  ".json": "application/json",
                }[ext] || "application/octet-stream";

              res.setHeader("Content-Type", contentType);
              fs.createReadStream(absPath).pipe(res);
            } else {
              res.statusCode = 404;
              res.end(`Not found: ${relModulePath}`);
            }
          });

          console.log(`📦 Dev alias active: ${baseUrlPath} → node_modules/*`);
        },
      },
    ],

    build: {
      minify: isMinify,
      lib: {
        entry: path.resolve(__dirname, "src/index.ts"),
        name: "spark",
        formats: ["es", "cjs"],
        fileName: (format) => {
          const base = format === "es" ? "spark.module" : `spark.${format}`;
          return isMinify ? `${base}.min.js` : `${base}.js`;
        },
      },
      sourcemap: true,
      rollupOptions: {
        // `three/webgpu` and `three/tsl` are externalized: the WebGPU backend
        // dynamically imports them at runtime so WebGL-only consumers never load
        // them. The consumer's import map / bundler must provide them when the
        // WebGPU path is used.
        external: ["three", /^three\/addons/, "three/webgpu", "three/tsl"],
        output: {
          globals: {
            three: "THREE",
          },
        },
      },
      emptyOutDir: isFirstPass,
    },

    worker: {
      rollupOptions: {
        treeshake: "smallest",
      },
      plugins: () => [
        glsl({
          include: ["**/*.glsl"],
        }),
      ],
    },

    server: {
      watch: {
        usePolling: true,
      },
      port: 8080,
    },

    optimizeDeps: {
      force: true,
      exclude: ["three", "three/webgpu", "three/tsl"], // prevent Vite pre-bundling
    },

    define: {
      sparkLocalAssets: localAssetsDirectoryExist,
    },
  };
});
