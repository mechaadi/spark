// Assemble a self-contained static site of ALL examples for hosting (e.g. on
// Vercel). The dev server serves three/lil-gui/stats via a node_modules alias
// middleware (see vite.config.ts) that does NOT exist on a static host, so here
// we materialize those vendored deps plus the built Spark bundle into an output
// directory that can be deployed as plain static files — no build step needed on
// the host (which matters because `npm install` there fails on the uncommitted
// `spark-rs` wasm dep).
//
// Usage:  npm run build            # produce a fresh dist/spark.module.js
//         node scripts/build-examples-site.js
//         cd examples-site && npx vercel        (or: --prod)
//
// Splat assets are NOT copied — examples fetch them from the external asset
// server via examples/assets.json, so the deploy stays small.

import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";

const OUT = "examples-site";
const VENDOR = path.join(OUT, "examples/js/vendor");

function copy(src, dest) {
  if (!existsSync(src))
    throw new Error(`missing: ${src} (run npm install / npm run build first)`);
  mkdirSync(path.dirname(dest), { recursive: true });
  cpSync(src, dest, { recursive: true });
}

console.log(`Assembling ${OUT}/ …`);
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });

// 1. All example folders, minus the heavy assets dir (loaded from the remote
//    asset server) and any previously-assembled vendor tree.
cpSync("examples", path.join(OUT, "examples"), {
  recursive: true,
  filter: (src) => {
    const parts = src.split(path.sep);
    return !parts.includes("assets") && !parts.includes("vendor");
  },
});

// 2. The built Spark bundle (examples import ../../dist/spark.module.js).
copy("dist/spark.module.js", path.join(OUT, "dist/spark.module.js"));

// 3. Vendored runtime deps (the dev-server alias targets), version-matched to
//    node_modules so they line up with the bundle.
copy("node_modules/three/build", path.join(VENDOR, "three/build"));
copy(
  "node_modules/three/examples/jsm",
  path.join(VENDOR, "three/examples/jsm"),
);
copy("node_modules/lil-gui/dist", path.join(VENDOR, "lil-gui/dist"));
copy("node_modules/stats.js/build", path.join(VENDOR, "stats.js/build"));

// 4. A root landing page linking to every example (single-page dirs link to the
//    folder; multi-page dirs link to each .html entry).
const entries = [];
for (const d of readdirSync("examples", { withFileTypes: true })) {
  if (!d.isDirectory() || d.name === "js" || d.name === "assets") continue;
  const dir = path.join("examples", d.name);
  const htmls = readdirSync(dir).filter((f) => f.endsWith(".html"));
  if (htmls.includes("index.html")) {
    entries.push({
      name: d.name,
      links: [{ label: d.name, href: `examples/${d.name}/` }],
    });
  } else if (htmls.length) {
    entries.push({
      name: d.name,
      links: htmls.sort().map((h) => ({
        label: h.replace(".html", ""),
        href: `examples/${d.name}/${h}`,
      })),
    });
  }
}
entries.sort((a, b) => a.name.localeCompare(b.name));

const rows = entries
  .map(({ name, links }) => {
    if (links.length === 1 && links[0].label === name) {
      return `    <li><a href="${links[0].href}">${name}</a></li>`;
    }
    const a = links
      .map((l) => `<a href="${l.href}">${l.label}</a>`)
      .join(" · ");
    return `    <li><span class="n">${name}</span> — ${a}</li>`;
  })
  .join("\n");

writeFileSync(
  path.join(OUT, "index.html"),
  `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spark — Examples</title>
  <style>
    body { margin: 0; padding: 2rem; background: #0e0e10; color: #e6e6e6;
      font: 15px/1.6 system-ui, sans-serif; }
    h1 { font-size: 1.4rem; }
    p { color: #9a9aa2; }
    ul { list-style: none; padding: 0; columns: 2; column-gap: 2rem; max-width: 720px; }
    li { break-inside: avoid; margin: .2rem 0; }
    .n { color: #fff; }
    a { color: #6cf; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Spark — Examples</h1>
  <p>${entries.length} examples. WebGPU pages need a WebGPU-capable browser.</p>
  <ul>
${rows}
  </ul>
</body>
</html>
`,
);

console.log(`Done. ${entries.length} examples → ${OUT}/`);
console.log(`Deploy:  cd ${OUT} && npx vercel`);
