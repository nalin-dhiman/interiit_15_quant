import { spawnSync } from "node:child_process";
import { cp, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

function run(command, args, cwd = process.cwd(), capture = false) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: capture ? "pipe" : "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw new Error(`${command} ${args.join(" ")} failed (${result.status})`);
  return capture ? result.stdout.trim() : "";
}

// Publishes only the static build, using a normal fast-forward branch update.
// Git handles authentication; no credentials are read or written by this tool.
const source = process.cwd();
const remote = run("git", ["remote", "get-url", "origin"], source, true);
const name = run("git", ["config", "user.name"], source, true);
const email = run("git", ["config", "user.email"], source, true);
run("npm", ["test"], source);
run(process.execPath, ["tools/build.mjs"], source);
const branch = "gh-pages";
const previous = run("git", ["ls-remote", "--heads", "origin", branch], source, true);
const directory = await mkdtemp(join(tmpdir(), "rfq-pages-"));
try {
  await cp(join(source, "dist"), directory, { recursive: true });
  run("git", ["init", "--initial-branch", branch], directory);
  run("git", ["config", "user.name", name], directory);
  run("git", ["config", "user.email", email], directory);
  run("git", ["remote", "add", "origin", remote], directory);
  if (previous) {
    run("git", ["fetch", "--depth", "1", "origin", branch], directory);
    run("git", ["reset", "--soft", "FETCH_HEAD"], directory);
  }
  run("git", ["add", "--all"], directory);
  const changes = spawnSync("git", ["diff", "--cached", "--quiet"], { cwd: directory });
  if (changes.status === 0) {
    console.log("The published static branch already matches this build.");
  } else if (changes.status === 1) {
    run("git", ["commit", "-m", "Publish Inventory Lab static application"], directory);
    run("git", ["push", "origin", `HEAD:${branch}`], directory);
  } else {
    throw new Error("Could not inspect the static publication changes.");
  }
} finally {
  await rm(directory, { recursive: true, force: true });
}
