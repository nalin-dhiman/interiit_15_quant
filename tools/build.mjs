import { cp, mkdir, rm, writeFile, readdir, stat } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { readFile } from "node:fs/promises";

await rm("dist", { recursive: true, force: true });
await mkdir("dist", { recursive: true });
await cp("web", "dist", {
  recursive: true,
  filter: (source) => !source.endsWith(".DS_Store") && !source.endsWith(".md"),
});
await writeFile("dist/.nojekyll", "");
const files = [];
async function walk(dir) {
  for (const name of await readdir(dir)) {
    const file = join(dir, name);
    if ((await stat(file)).isDirectory()) await walk(file);
    else files.push(file);
  }
}
await walk("dist");
for (const required of [
  "dist/index.html",
  "dist/app.js",
  "dist/engine/index.js",
]) {
  if (!files.includes(required))
    throw new Error(`Missing required application asset: ${required}`);
}
const manifest = {};
for (const file of files.sort())
  manifest[file.slice(5)] = createHash("sha256")
    .update(await readFile(file))
    .digest("hex");
await writeFile(
  "dist/build-manifest.json",
  JSON.stringify(manifest, null, 2) + "\n",
);
console.log(
  `Built ${files.length} static assets. No manuscript, SDK, credentials or raw experiment files enter dist/.`,
);
