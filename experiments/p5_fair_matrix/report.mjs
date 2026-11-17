import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const EXP_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_DIR = dirname(dirname(EXP_DIR));
const VENV_PY = join(REPO_DIR, ".venv", "bin", "python");
const PY = existsSync(VENV_PY) ? VENV_PY : "python3";

execFileSync(PY, [join(EXP_DIR, "report.py")], {
  cwd: REPO_DIR,
  stdio: "inherit",
});
