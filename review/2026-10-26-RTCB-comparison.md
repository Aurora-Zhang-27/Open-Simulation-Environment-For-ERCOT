# RTCB-Gym 与两个相关项目的代码对比
日期：2026-10-26。审查目标：判断继续建设独立 post-RTC+B 储能策略环境是否有价值，以及下一步最小验证应当是什么。

## 判断

**有条件值得继续，但先验证一个有限的研究问题。** 现有代码值得保留；目前不足以支撑“第一个 RTC+B 仿真器”或“已完整实现新市场机制”的定位。继续投入的理由，应当是规则、信息时序、物理可行性和评价标准的一致性能够改善方法评价，而不是单凭现有项目数量少。

两个相关仓库均属于同一 GitHub 开发者。ercot-rtcb-bench 侧重数据、优化与回测，HybridBid 侧重具体学习方法实验。它们已覆盖不少工作，但本次发现的实质问题说明“可用、可信、可扩展的参考环境”仍值得检验。发现别人的实现问题不自动构成论文贡献，也不意味着这些项目的全部结果无效。

## 审查范围和证据等级

固定提交：
- RTCB-Gym：`7586e31095666d7c8e5cc59c7a88b47e73e5c058`。
- ercot-rtcb-bench：`8eb6c343340184bdd5d0491e9d47d3cf1310a32c`。
- HybridBid：`68c1a62398e53ed4785c4dd67996ed547c4da34b`。

通过 GitHub 连接读取固定提交的源码、测试、配置、数据说明、结果及决策记录；在临时目录保存源文件并运行测试和合成诊断。没有修改三个仓库的实现，没有训练大型模型，没有独立重新下载全部历史数据或重跑三家的真实市场收益表。

证据分为：实际运行得到的合成反例、源码直接确认的行为、由这些行为推导的局限、尚需真实数据实验验证的影响。下文不会把合成反例当成“历史收益已被高估多少”的证据。

## 功能对照

| 方面 | RTCB-Gym | ercot-rtcb-bench | HybridBid |
|---|---|---|---|
| 主要用途 | 通用策略接入、能量交易环境与结算 | RTC+B 数据、LP/随机优化/DFL 回测 | 储能离线学习和预测优化实验 |
| 时间步 | 15 分钟 SPP 结算回放 | 5 分钟 LMP/MCPC；主要滚动回测每小时提交一段决策 | 5 分钟；Gym 与最终回测是不同路径 |
| 能量与辅助服务联合动作 | 尚未实现辅助服务 | 已有能量与五类 AS 变量 | 已有能量与五类 AS 动作 |
| Gymnasium | 已实现、统一 Policy 接口 | envs/ 只有包占位文件 | 已实现，但训练与评价契约不同 |
| 日前与实时双结算 | 已实现简化版日前头寸与偏差结算 | 本次检查的核心回测将 DAM 作为预测输入，不是双结算头寸 | 本次检查的核心回测将 DAM 作为特征，不是双结算头寸 |
| 信息时间线 | RT 严格过去窗口；DA 单独决策 context | analog 选择会读取目标日整日实际净负荷 | observation 包含同一索引的 RT 价格，随后以该价格结算 |
| 物理与 oracle 一致性 | 能量模型已有 LP 检查/MILP 互斥、环境回放校验 | LP 允许同时充放电；AS 持续时间约束主要只有 Non-Spin | 训练投影、评价投影、优化器存在不同定义 |
| 连续状态 | RT 支持多日 episode；双结算目前一天一集 | rolling_lp 在面板内持续传递 SoC | Gym 日初重置，最终 T-60 评价连续传递 SoC |
| 数据与复现 | 两个 hub、较长时期、hash、缺口报告；数据未附 | 版本化数据 DOI、数据校验、方法和修订记录比较充分 | 附带数据与模型产物；目前部分入口有旧目录导入问题 |
| 市场出清与执行 | 历史价格接受者、日前计划全额授予 | 历史价格接受者，直接优化奖项数量 | 历史价格接受者，直接给容量计费；评价不模拟 AS 实际部署 |
| 实际市场验证 | 仍缺独立数据复现和真实执行/结算对账 | 数据与诊断较丰富，不能据此认定完整协议一致性 | 有结果文件，不能据此认定可部署或独立样本效果 |

以上“尚未实现”限定于对应固定提交和本次核对的运行路径。范围小与实现错误是不同问题。

## ercot-rtcb-bench：已有内容与缺口

### 应当认可的工作

1. 能量、RegUp、RegDown、RRS、ECRS、Non-Spin 的联合优化接口已存在；确定性、情景均值、两阶段随机优化、DFL 都有实现。你的项目若只增加同样的动作维度，差异不足。
2. 数据版本、DOI、数据校验、结果重算和负结果记录已经较充分。不能把“加数据说明、加基线、加复现脚本”当作它完全没有做的事。
3. 面板内滚动 SoC 已存在，跨日连续状态本身不是你的独占缺口。
4. Gym 和库级评估包确实仍为占位，见 [envs](https://github.com/karthikmattu06-hue/ercot-rtcb-bench/blob/8eb6c343340184bdd5d0491e9d47d3cf1310a32c/src/ercot_rtcb_bench/envs/__init__.py) 与 [README](https://github.com/karthikmattu06-hue/ercot-rtcb-bench/blob/8eb6c343340184bdd5d0491e9d47d3cf1310a32c/README.md)。不过 HybridBid 已有 Gym，所以“做一个 Gym wrapper”仍不够。

### 已复现：负价格下 oracle 为物理松弛

[perfect_foresight.py](https://github.com/karthikmattu06-hue/ercot-rtcb-bench/blob/8eb6c343340184bdd5d0491e9d47d3cf1310a32c/src/ercot_rtcb_bench/methods/perfect_foresight.py) 使用独立充电 c、放电 d 变量，没有互斥约束。源码和决策记录认为同时充放电次优，但这个判断在负电价下不普遍成立。

合成输入：10 MW / 20 MWh、RTE 0.85、初始满电、单个五分钟时段、LMP=-100、美金计价、AS 价格全为零。

实际解：
- 充电 10 MW，同时放电约 8.5 MW；
- 净购电约 1.5 MW，SoC 保持满电；
- 收益约 $12.50。

在同样的单净功率电池定义下，满电无法继续净充电，负价格放电又亏钱，最优是闲置，收益 $0。RTCB-Gym 在同一例子中切到 MILP，返回闲置和 $0。

这并不意味着该 LP 不能成为上界：它仍可能是一个松弛上界。问题是它不能被无条件解释为单净功率物理电池的可达 perfect-foresight 收益；capture rate 可能因放松约束而偏低。本次未量化它对历史面板结果的影响。

### 已复现：部分 AS 奖项缺少能量可交付性

同一 LP 对 Non-Spin 有四小时 SoC 约束，但 RegUp、RRS、ECRS 没有对应持续时间/库存约束。

合成输入：初始 SoC=0、能量价格=1、RegUp=1000，其余价格零。优化器选择充放电均为零，同时授予 10 MW RegUp，收入 $833.33/五分钟。对于无储能、无充电基线可削减的状态，这不是可持续提供上调能量的承诺。

这说明它的可行域需要对照协议补充，不能仅因变量覆盖了五类 AS 就认定规则已完整实现。其随机 LP 和 DFL 层也沿用了类似的功率共享与 Non-Spin SoC 结构。

### 已复现：目标日实际净负荷影响 analog 选择

[analog.py](https://github.com/karthikmattu06-hue/ercot-rtcb-bench/blob/8eb6c343340184bdd5d0491e9d47d3cf1310a32c/src/ercot_rtcb_bench/forecaster/analog.py#L113) 对目标日调用 get_net_load(target_day)，再以完整日曲线选历史相似日。[data_loader.py](https://github.com/karthikmattu06-hue/ercot-rtcb-bench/blob/8eb6c343340184bdd5d0491e9d47d3cf1310a32c/src/ercot_rtcb_bench/forecaster/data_loader.py#L115) 返回的是目标日 net_load_mw，不是按决策时点保存的预测版本。

我保持目标日前半天完全不变，只改后半天净负荷，所选 analog 从 2026-01-10 变成 2026-01-11。这确认选择依赖未来部分。

若任务是目标日开始前或日内的因果决策，目标日尚未实现的实际净负荷不应进入信息集。回测脚本调用这个 forecaster，因此“只用了过去 analog 日”本身不足以保证无前视。需要将目标曲线换成当时已发布的预测，并固定发布时间。本次没有计算纠正后真实收益会变化多少。

### 复现上的边界

- 本次原有测试实际结果：97 passed、14 skipped。跳过包括未提供的真实数据文件相关检查。合成反例能与已有测试同时成立。
- [stochastic_lp.py](https://github.com/karthikmattu06-hue/ercot-rtcb-bench/blob/8eb6c343340184bdd5d0491e9d47d3cf1310a32c/src/ercot_rtcb_bench/methods/stochastic_lp.py#L49) 直接 import gurobipy 并建模。通用 solver 的 HiGHS/SciPy fallback 不自动覆盖这个随机优化路径。
- DAM 已知时间、目标日整日曲线、数据修订版本等都需要统一的 as-of 机制；这些比一般的数据 train/test 日期切分更细。

## HybridBid：已有内容与缺口

### 应当认可的工作

已有 Gym、能量与 AS 联合动作、可行性投影、学习算法、专家轨迹、连续 SoC 评价和结果文件，部分未成功的方法也有关闭记录。这是明确的相关工作，不能再声称“没有人做过 RTC+B 学习环境”。

但它目前更接近具体实验工程。训练 Gym、专家优化和最终评价之间的定义并未完全统一，因此不能仅凭 README 的成绩判断它已经解决了通用仿真环境的问题。

### 官方规则对照：最终评价的持续时间参数不符

[最终评价](https://github.com/karthikmattu06-hue/hybridbid/blob/68c1a62398e53ed4785c4dd67996ed547c4da34b/src/evaluation/eval_t60.py) 采用：
RegUp 1h、RegDown 1h、RRS 1/6h、ECRS 1/4h、Non-Spin 0.5h。

ERCOT [NPRR1282](https://www.ercot.com/mktrules/issues/NPRR1282) 明确于 2025-12-05 生效：Regulation 和 RRS 持续时间更新为 30 分钟，ECRS 更新为一小时。2026 年的 [NPRR1339](https://www.ercot.com/mktrules/issues/NPRR1339) 进一步说明，ECRS as-built 要求是一小时，部分旧文本仍写两小时，需要清理。NPRR1339 页面当前仍显示待批准；这里引用的是它对已实施要求的解释，不把待批准条文当作已生效新规则。

所以代码注释“ERCOT-correct”不能作为参数正确性的证据。

运行同一个 project_action：
- SoC=4 MWh，最低 SoC=2 MWh，效率 0.95，申请 10 MW ECRS；
- 当前 0.25h 参数允许约 7.6 MW；
- 仅将其自身公式的持续时间改成 1h，容量上限为 1.9 MW。

这是具体的四倍容量差异，不是格式或命名问题。它会影响可行域；是否改变历史策略排名仍需重跑。

### 训练和评价是两个不同的任务

- [训练投影](https://github.com/karthikmattu06-hue/hybridbid/blob/68c1a62398e53ed4785c4dd67996ed547c4da34b/src/models/feasibility.py)：持续时间表为 0.5、0.5、0.5、1、4 小时，并且对所有 AS 采用另一套库存扣减/缩放逻辑。
- [最终评价投影](https://github.com/karthikmattu06-hue/hybridbid/blob/68c1a62398e53ed4785c4dd67996ed547c4da34b/src/evaluation/eval_t60.py#L215)：使用上面的另一套持续时间、上下调不同库存条件和总容量共享约束。
- 同一 SoC=4、10 MW ECRS 输入，训练投影接受 2 MW，评价投影接受约 7.6 MW。这已实际运行确认。
- [训练 Gym](https://github.com/karthikmattu06-hue/hybridbid/blob/68c1a62398e53ed4785c4dd67996ed547c4da34b/src/env/ercot_env.py) 的 reward 包含按归一化能量动作计算的 energy_term、EMA timing bonus 和按实际 MW 计的 AS 收入；[最终评价](https://github.com/karthikmattu06-hue/hybridbid/blob/68c1a62398e53ed4785c4dd67996ed547c4da34b/src/evaluation/eval_t60.py) 用实际功率乘价格计费。reward shaping 可以是合理训练手段，但这两者不应被混称为同一个美元收益契约。
- Gym 每天重置 50% SoC；最终评价整个窗口连续传递 SoC。作者自己也记录了这个分布差异。

### 同期价格可见：决策任务需要重新定义

最终评价的 _build_obs 使用 price_data[start:idx+1]，策略看完 observation 后，收益再用 price_data[idx] 计算。合成输入中，我将 idx=31 的结算价格设为 1234，策略 observation 最后一行立即包含 1234。

这与 RTCB-Gym 的 t-1 截止信息集不同。如果目标是先报价后出清，这构成同期结果信息优势；如果定位为已知 SCED 信号后的执行控制，就必须明确发布/执行时点，并说明历史价格信号与结算价格的对应关系。当前代码没有提供足够的事件时间契约来把两者视为同一任务。

### 训练/评价窗口不能当作独立样本外验证

[专家轨迹生成](https://github.com/karthikmattu06-hue/hybridbid/blob/68c1a62398e53ed4785c4dd67996ed547c4da34b/src/data/postbreak_milp.py) 定义 train=2025-12-05..2026-02-10，val=2026-02-11..2026-04-15；[BC 入口](https://github.com/karthikmattu06-hue/hybridbid/blob/68c1a62398e53ed4785c4dd67996ed547c4da34b/src/methods/bc/run_bc.py) 加载相应 train/val 文件。最终 T-60 评价为 2026-01-01..2026-02-23，整个评价窗口落在这两个窗口内。

这足以说明公开配置不提供与 train/val 分离的独立测试窗口。它不证明具体模型记忆了某天，也不能据此推断精确过拟合量。还需检查实际轨迹版本和重跑严格 held-out 评价才能量化。

### 当前入口有发布整理问题

BC 入口仍 import experiments.prepare_postbreak，但固定提交的文件树没有该模块；对应评价已移动到 src/evaluation/eval_t60.py。运行入口导入即得到 ModuleNotFoundError: No module named 'experiments'。ROOT=parents[2] 还指向 src，和部分 data/checkpoint 路径不匹配。

这可能来自仓库整理，并不反证历史实验从未运行；但它确实妨碍外部用户按当前 README 直接复现。单纯修复这些导入问题是工程修复，不足以构成你的研究差异。

### 本次测试边界

不依赖市场数据和未安装优化依赖的三个原有测试文件：35 passed、2 skipped。没有把这个数字称为全仓库测试通过；没有重跑 BC 训练及 T-60 全部真实轨迹。

## RTCB-Gym：你已有的价值与仍缺的内容

### 目前可以保留的优势

1. [Policy](https://github.com/Aurora-Zhang-27/Open-Simulation-Environment-For-ERCOT/blob/7586e31095666d7c8e5cc59c7a88b47e73e5c058/rtcb_gym/baselines/base.py) 接口与 [run_policy](https://github.com/Aurora-Zhang-27/Open-Simulation-Environment-For-ERCOT/blob/7586e31095666d7c8e5cc59c7a88b47e73e5c058/rtcb_gym/evaluation.py) 统一，策略比较更容易接入和审计。
2. [RT observation](https://github.com/Aurora-Zhang-27/Open-Simulation-Environment-For-ERCOT/blob/7586e31095666d7c8e5cc59c7a88b47e73e5c058/rtcb_gym/env.py) 截止到上一时段；[DA context](https://github.com/Aurora-Zhang-27/Open-Simulation-Environment-For-ERCOT/blob/7586e31095666d7c8e5cc59c7a88b47e73e5c058/rtcb_gym/day_ahead.py) 独立限制日前决策信息。这里的优势是代码定义明确，尚不等于已验证所有实际数据发布延迟。
3. 物理模型、实际执行功率、结算和 oracle 回放已有一致性测试；负价格同时充放电在 [优化器](https://github.com/Aurora-Zhang-27/Open-Simulation-Environment-For-ERCOT/blob/7586e31095666d7c8e5cc59c7a88b47e73e5c058/rtcb_gym/baselines/optimize.py) 中有专门处理。
4. 两个 hub、时间切分、数据 hash、缺口、DST、退化成本、结算拆分已有基础。
5. 日前头寸与偏差的双结算是这三条主要运行路径中你已落实的独立功能；对方使用 DAM 特征不等于已经完成同样的双结算。

### 目前不能声称的内容

1. 还没有辅助服务动作、持续时间、可交付库存、能量与 AS 联合可行域。现在是 post-RTC+B 数据上的能量交易环境，不能据此声称完整实现 RTC+B 的核心调度机制。
2. 15 分钟结算回放不是五分钟报价/调度。可以保留 settlement 层，同时另建五分钟 decision/execution 层；不能简单把 15 分钟值复制三次视为真实五分钟数据。
3. DA schedule 是自动授予的可行计划，没有价格-数量 offer、成交选择、部分中标、资源资格或实际出清过程。
4. 没有 AS 调用/部署能量对 SoC 的影响，以及履约/未履约等机制。未来若缺实际部署数据，可做明确标识的情景假设和敏感性分析，不应把生成部署当作真实部署。
5. 双结算强制 episode_days=1，尚无连续日前—实时库存衔接。只把 soc_init 与 soc_terminal_min 都设为 0.5，并不把前日实际库存传到次日。
6. 尚无规则版本到代码约束、再到测试案例的映射，也未用真实储能执行/结算数据验证外部有效性。
7. 单元测试和 optimizer replay 证明实现内部性质，不能单独证明市场规则正确、策略可部署或收益真实可获得。
8. 原路线图的 stress generator 与 LLM agent 现在优先级较低；它们不解决上述核心问题。

### 本地测试

在临时副本、Python 3.13、pandas 2.3.3、Gymnasium 1.2.3 下，151 passed。首次使用本机 pandas 3.0.2 与 PyArrow 19 组合时，出现三项 parquet 时区读取失败（148 passed）；换为临时目录中的 pandas 2.3.3 后全部通过，源码未改。应把它记为依赖组合兼容性边界，不能混同于已证实的结算模型错误。线上 Python 3.10/3.12 的 CI 在前一轮也已确认通过。

## 真正仍有空间的工作

优先验证以下组合是否尚未被这些固定提交充分覆盖：

- 按明确生效日期选择规则参数，并把每条关键规则对应到可审查的约束与反例测试。
- 区分决策时间、信息发布时间、调度/执行时间和结算时间，支持 as-of 数据快照；未来数据变动不能改变当时 observation 或合法策略输出。
- 策略、oracle、投影和环境共享物理可行域与收益账本，避免训练和评价各写一套规则。
- 在该统一环境中提供五分钟能量与所选 AS 的策略评测，并对日前头寸与实时运行做明确的边界处理。
- 用真实 held-out 市场窗口验证规则差异会不会改变容量承诺、收益偏差或策略排名，再给出跨 hub / 电池时长 / 稀缺时段的稳健性边界。

以上是“值得测试的差异定位”，不是已经完成的文献空白证明。ERCOT 自身也公开过 [RTC+B Simulator V3](https://www.ercot.com/committees/inactive/rtcbtf)；本次仅确认其存在，未审计该 Excel 工具的覆盖范围，因此也不应声称所有官方仿真都不存在。

## 建议的继续/停止门槛

### 第一关：先完成一条规则链

先做能量 + ECRS 的最小版本：用已确认的一小时持续要求，明确 SoC 与可提供容量的关系、五分钟信息时序、收益账本和 AS 不部署/部署的假设边界。用同一物理内核运行规则策略、MPC 和 oracle，并重放 oracle。

之所以先选一个产品，是为了尽早检验规则到仿真的全过程。不要先把五类 AS 都铺开，再发现定义或数据时序错误。

通过条件：规则有官方来源和生效日期；每个约束有边界案例；oracle 与环境回放一致；未来数据扰动不改变既有决策信息；同一合法动作不因训练/评价路径不同而得到不同结果。

### 第二关：量化“正确建模”的实际影响

在预先固定的真实 held-out 窗口，比较旧/简化约束与核实后的约束，记录：
- 不可交付 AS 承诺频率与幅度；
- 纠正后的收益与风险变化；
- 策略排名是否改变；
- 这些变化是否仅由一个极端日驱动。

合成反例只能通过第一关，不能替代这一步。不要把已看过并据此选过模型的窗口重新称为 untouched test。

### 第三关：证明它可以作为公共工具使用

让不同方法通过同一 evaluate 入口运行；将数据、规则配置、软件环境和结果版本绑定；在独立干净环境复现一个小型市场窗口。另一个人/机器的复现尚未完成，不应提前写成事实。

### 何时继续

如果规则核对完成，真实样本中也显示简化模型导致不可交付承诺、实质收益偏差或策略判断改变，而且统一环境能稳定复现，则值得扩展到更多 AS、双结算连续运行和更广的数据。

### 何时收窄或停止大规模投入

- 如果最终增量只有 wrapper、更多价格日期或修复旧导入，而没有任务/规则/评价上的明确改进，应保留为工程作品或向现有项目贡献，不宜据此宣称独立研究突破。
- 如果无法获取足够的数据或核实关键规则，应明确停在受限模型，不扩展成“完整市场模拟器”。
- 如果修正建模对目标研究问题的结论基本无影响，可以发布小而清楚的工具/审计结果，但不应靠继续加算法来制造必要性。
- 不建议从零复刻全网 SCED、市场参与者行为和内生价格形成，除非未来有明确数据、合作方和研究需求支持。

## 收益比较禁区

三家表里的美元收益或 capture rate 不能直接排名：电池功率与时长、效率处理、退化成本、市场产品、hub/节点、时期、SoC 边界和 oracle 可行域均不同。必须先把方法移植到同一任务和同一评价协议中，才能声称哪个方法更好。

## 复核产物

- 同目录 evidence/probe_results.json：合成运行的原始数值。
- 同目录 evidence/rtcb_audit_probes.py：诊断脚本，输入参数为固定提交源码快照根目录。
- 本次快照根目录：/private/tmp/rtcb-audit-20261026，包含 rtcb-gym/、ercot-rtcb-bench/、hybridbid/。
- 原仓库单测命令与结果见 evidence/test_summary.json。

临时快照可能被系统清理；长期复现时请按上文三个 commit 获取源码，摆成相同目录，并安装对应依赖。诊断脚本没有修改源文件，也不下载数据。

