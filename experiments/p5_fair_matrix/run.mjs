import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const EXP_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_DIR = dirname(dirname(EXP_DIR));
const VENV_PY = join(REPO_DIR, ".venv", "bin", "python");

function ensureVenv() {
  if (!existsSync(VENV_PY)) {
    execFileSync("uv", ["sync", "--frozen"], { cwd: REPO_DIR, stdio: "inherit" });
  }
}

const [, , cmd, arg] = process.argv;
if (cmd === "list") {
  ensureVenv();
  const out = execFileSync(VENV_PY, [join(EXP_DIR, "run_unit.py"), "list"], {
    cwd: REPO_DIR,
    encoding: "utf-8",
  });
  process.stdout.write(out);
} else if (cmd === "unit" && arg) {
  ensureVenv();
  execFileSync(VENV_PY, [join(EXP_DIR, "run_unit.py"), arg], {
    cwd: REPO_DIR,
    stdio: "inherit",
    env: {
      ...process.env,
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || "1",
      OPENBLAS_NUM_THREADS: process.env.OPENBLAS_NUM_THREADS || "1",
      MKL_NUM_THREADS: process.env.MKL_NUM_THREADS || "1",
    },
  });
} else {
  console.error("Usage: node run.mjs <list|unit <id>>");
  process.exit(1);
}
