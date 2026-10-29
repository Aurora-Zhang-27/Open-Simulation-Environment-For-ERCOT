# 本地工作目录

本轮起在 `/Users/aurorazhang/Desktop/Open Simulation Environment/` 继续。
目标与范围以 [BLUEPRINT.md](BLUEPRINT.md) 为准。

## 来源

- 上游：[RTCB-Gym](https://github.com/Aurora-Zhang-27/Open-Simulation-Environment-For-ERCOT)。
- 源码基线：`7586e31095666d7c8e5cc59c7a88b47e73e5c058`，v0.2.0。
- 已带入本轮五分钟 ECRS 原型、测试、审查和探索结果。
- 已带入用于探索的第三方公开处理后价格，位于 git 忽略的
  `data/raw/third_party_hybridbid/`；这不是云端原始数据。
- `review/workspace_transfer_manifest.json` 记录本次文件迁移与校验。
- 原蓝图已备份至 `archive/blueprints/`，旧 Documents 工作区保留。

本目录是本地文件工作区；没有复制旧工作区的 `.git`、缓存或临时 Python
依赖。上游基线和本地增量通过清单、代码哈希及报告保留来源。

## 常用入口

```bash
cd '/Users/aurorazhang/Desktop/Open Simulation Environment'
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/run_ecrs_gate.py
```

数据请求：[云端导出清单](docs/request_ercot_gate_data_zh.md)。
模型边界：[ECRS gate](docs/ecrs_gate.md)。
结果解释：[首轮验证](review/2026-10-26-ECRS-gate.md)。

当前目标是给别人使用的 CPU 工具，验收标准是 clone、安装、真实数据回放和账本核对。
自己训练 agent 暂停；既有探索的收益大小不再作为工具是否继续的门槛。
见 [快速开始](docs/quickstart.md) 和更新后的蓝图。

新增离线示例：`python -m rtcb_gym.demo --out results/demo`。
带入导出提交 `c86ae8d` 的代码并作本地修订：默认离线，修复价格沿用时限的部分覆盖边界。
导出代码已到本机；实际云端 CSV、报告和缓存仍未到本机。
