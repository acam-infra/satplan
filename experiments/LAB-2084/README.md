# LAB-2084 · 强静态与无争用接续的需求验证

在 `acam-infra/satplan@583b8337603c` 上构造资源窗口驱动的 6 星混合请求主场景，配对对比 `Static`（地面可见预测 + `orbit_thermal` 强单次分配）与 `Dyn-CF`（重放同一初始分配，无多流链路争用，保留单笔 KV 传输、传播、暂停恢复与热/电约束）。

## 资源窗口与单迁移时间预算
- **源星窗口（`CS-1..3`）**：日照 `[0, 85]s`（`300W, f=1.0`, `~72.7 tok/s`）可解码约 `6,000 tok`；地影 `[85, 210]s` 受 `eclipse_dod_budget_j = 1000 J` 限制，仅够 `20W (f=0.35)` 运行 `50s`（`[85, 135]s`, `~2,140 tok`），随后 `0W` 停机 `75s` 至 `t=210s`。长请求（`pout = 9500..10800`）在出影后叠加 `T_max = 85°C` 热限频，完成时间落于 `230–310s`。
- **邻星窗口（`CS-4..6`）**：日照 `[90, 220]s`（`130s` 连续高功率窗口）。
- **单笔迁移预算**：`bpt = 128 KiB/tok`，`t=90s` 时约 `6,500–7,000 tok` KV（`~850–915 MB`），经 `0.6 Gbps (75 MB/s)` 1 跳星间链路 + `5 ms` 传播耗时 `~11.3–12.2 s`，剩余 `~3,500 tok` 在邻星需 `~48 s`，总完成时间 `~150–175 s < deadline (+195..+255s)`，满足「源星预计完不成，转移开销 + 剩余解码仍赶得上」。

## 复现命令
```bash
# 1. 全量单元测试与 LAB-2084 回归测试
PYTHONPATH=. uv run pytest -q

# 2. 在 cnb-compute2 (16 核) 运行锚点复现 + Dev(101-103) + Val(142-146, 6星 & 24星)
PYTHONPATH=.:src uv run python experiments/LAB-2084/run_exp.py --phase all --workers 16

# 3. 从原始 JSON 生成汇总表与 CDF / 配对差值图
PYTHONPATH=.:src uv run --with matplotlib==3.10.9 --with numpy python experiments/LAB-2084/analyze_and_plot.py
```

## 产出文件 (`experiments/LAB-2084/out/`)
- `anchor_repro.json`: LAB-2082 6 星 seed 142 与 24 星 5 种子复现锚点
- `dev_6sat.json`: 开发种子 `101–103` 扫描与静态策略对比 (`orbit_thermal` vs `least_loaded` / `nearest`)
- `val_6sat.json` / `val_24sat.json`: 独立验证种子 `142–146` 配对结果
- `sample_logs_seed142.json`: seed 142 初始映射、事件日志与迁移决策账本
- `summary_tables.md`, `cdf_long_short_val.png`, `paired_delta_grid.png`, `manifest.json`
