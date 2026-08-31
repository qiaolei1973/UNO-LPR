# UNO-LPR 问题清单

基于 2026-08-31 的代码审计、实测运行与论文（UNO_AAAI2026.pdf，AAAI-26）交叉验证。每个块内独立编号。每条含三个字段：**问题** / **修复方式**（不可修时说明原因与能否自行设置）/ **影响**（修复或设置后可能带来的偏差或错误）。E2 为评估结论，不套三字段。

## A. 运行阻塞（代码无法启动/空转）

### A1. 悬空导入 `POMDPDKT`

**问题**：`Agent/genRec.py:6`。`POMDPDKT → UniLPR` 重命名遗漏，类从未定义。`import Agent` 即抛 `ImportError`，所有 agent 训练入口全挂。

**修复方式**：删除该导入（已修，commit 966e17f）。

**影响**：无副作用——纯恢复 import，不改变任何语义，是复现的前提。

### A2. DKT 预训练保存必挂

**问题**：`LifeLongKT/trainKT.py`。`os.makedirs(os.path.dirname(model_save_path))` 只建 `saved_models/DKT`，缺 `junyi/` 末层，`torch.save` 抛 `Parent directory ... does not exist`。实测复现。

**修复方式**：改 `os.makedirs(model_save_path, exist_ok=True)`，一行。

**影响**：无副作用；修复后 DKT 预训练才能产出 checkpoint。不改则永远无法得到训练好的 DKT，env 只能随机初始化。

### A3. `xxx.pth` 占位符 + 静默随机 DKT

**问题**：`trainLPR.py setup_environment` 硬编码 `model_path = './LifeLongKT/saved_models/DKT/junyi/xxx.pth'`（字面量，初始 commit 就有）。trainKT 实际产出 `<随机数>_seed_42_best.pth`，文件名不匹配；文件缺失时静默回退随机 DKT → reward 恒 0、actor_loss 恒 0，训练完全空转（实测）。

**修复方式**：glob 最新 checkpoint + 缺失时 `raise FileNotFoundError`。

**影响**：① glob 按文件名排序选"最新"，若目录有多个 checkpoint（不同 epoch/seed），选到的未必是论文所用的那一个 → DKT 权重不同 → env 模拟与奖励分布不同 → 数字偏差；② 改显式报错后，无 checkpoint 时训练直接中断（不再"假装能跑"），这是正确行为但要求先跑通 trainKT。

### A4. 默认入口损坏

**问题**：`trainLPR.py` hydra `config_name="Random"`，但 `initialize_agent` 无 Random 分支 → AttributeError。

**修复方式**：hydra 默认 config 改 `UNO`。

**影响**：无副作用；`python trainLPR.py` 从"必崩"变为正常跑 UNO 训练，与 README 命令一致。

### A5. GRPO.yaml 死配置

**问题**：`config/{junyi,assist09}/GRPO.yaml` 存在但从未有 GRPO agent（GRPO 目标内联在 `UNO.py:learn()`）。

**修复方式**：删除或补实现。

**影响**：删除无行为变化（从未被使用）；补实现则新增一个论文未描述的实验入口，且与 UNO 内联的 GRPO 目标并存，口径需对齐否则容易误用。

## B. 论文 ↔ 代码协议不一致

### B1. 阈值语义不一致

**问题**：env（`Env_junyiDKT.py`）用 DKT `states=True` 的**原始 logit** 判掌握/答对：`logit > 0.5`（⟺ `sigmoid > 0.622`）；trainKT 评估 ACC 用 `sigmoid(logit) > 0.5`（⟺ `logit > 0`）。env 阈值更严，奖励分布系统性偏移。

**修复方式**：env 判掌握改 `sigmoid(x) > 0.5`（一行），阈值可自行设置。

**影响**（关键）：改口径后初始分、答对判定、reward 全部系统性变化 → 所有方法 Ep 数值变化，**无法确定论文数字对应哪个口径**；不改则维持现状。修复本身不报错，但会让结果与论文表格的对账更加不确定。

### B2. UniLPR 预训练缺失

**问题**：论文："Before generating recommendations, we train the model using λ₁L_KL + λ₂L_KS"（Eq 16/17 双监督预热）。代码：`UNO.__init__` 随机初始化 `UniLPR(...)`，无任何 load_state_dict；`begin_episode` 仅每 episode 做一个监督步，预训练产物与主流程脱节。

**修复方式**：trainKT 产出 checkpoint 后让 UNO 加载；预训练轮数可自行设置。

**影响**：① 加载后模型起点改变 → 收敛与最终 Ep 变化；② 论文未给预训练细节（epoch/lr/划分），自行设置大概率与作者实际协议不同 → 数字偏差；③ **数据泄漏风险**：trainKT 用 80/10/10 学生划分预训练，与 RL 训练/测试的学生划分可能重叠，预训练若见过测试学生，测试数字会被污染（偏乐观）。

### B3. EM/RM 双模型缺失

**问题**：论文：环境模型 EM + 奖励模型 RM 两个预训练 DKT，RM 测试时停用。代码：只加载**一个** DKT，模拟与状态同源，无停用逻辑（`outside_model = None` 注释暗示作者曾计划/使用第二模型）。

**修复方式**：改 env 加载第二个 DKT + 测试停用逻辑；或自行选择接受单模型近似。

**影响**：① 实现"测试时停用 RM"需切换测试阶段的观察来源，改动面较大，做错会改变行为；② 单模型近似下"模型既模拟又评价"，存在自证偏差（奖励来自模拟自身的状态），与论文双模型口径不同；③ 第二个 DKT 的 checkpoint 论文未提供，只能自训，其质量直接决定奖励信号。

### B4. 数据划分不一致

**问题**：论文：85/15 训练/测试；代码 `trainKT.py`：80/10/10（README 同）。

**修复方式**：改 `train_ratio`。

**影响**：改 85/15 后 DKT 预训练数据量变化 → DKT 质量 → env 模拟与奖励分布 → 最终数字变化。代码与 README 一致但论文写 85/15，改不改都不确定哪个对应论文数字。

### B5. KT 训练 epoch 不一致

**问题**：README：100 epochs；`DKT.yaml`：`num_epochs: 50`。论文未给具体值。

**修复方式**：改 yaml，epoch 可自行设置。

**影响**：epoch 决定 DKT 收敛质量 → env 模拟真实性 → 数字变化。README 100 与代码 50 矛盾、论文未给，任何取值都是猜测；训不够 DKT 欠拟合（logits 贴近 0，reward 趋 0），训多有过拟合风险。

### B6. Case study 死配置

**问题**：`UNO.yaml` 有 `run_case_study: true`，但 `trainLPR.py` 从未读取；论文 Figure 5（学生 29144 案例）不可从仓库复现。

**修复方式**：需实现轨迹保存与出图（工作量中等）；不可自行设置。

**影响**：实现需在训练循环加轨迹保存，不动训练逻辑本身；产出 Figure 5 类案例图，与数值无关。

### B7. rct 目标数表述不符

**问题**：README："只有 1–5 target nodes"；代码 `rct-5/10/20` 设 `num_goals: 500`（作为上限，实际=去重后目标数）。

**修复方式**：改配置；轻微。

**影响**：实际目标数 = min(500, 去重后目标数)，500 只是上限；若改小（如 10）会改变任务难度与 Ep 量级 → 数字变化。

### B8. README 的 DKT ACC/AUC 表论文没有

**问题**：README 报 Junyi 0.8217/0.8475、Assist09 0.7328/0.7703；论文正文（含附录提取文本）无此表。

**修复方式**：不可修（论文/README 表述问题，非代码）；可自行用仓库数据重算核对。

**影响**：引用 README 数字需自行验证，无法与论文对照。

### B9. README 表格 trian/final 行论文没有

**问题**：论文 Table 2 只报测试（eval）值；README 额外加训练阶段 trian/final 行（来源应为训练日志）。

**修复方式**：不可修（表述差异）；可自行从训练日志复现。

**影响**：无（纯表述差异，不影响任何数值语义）。

## C. 复现性障碍

### C1. 种子未固化

**问题**：全仓库仅一行 `random.seed(seed)`（`StudentDataload.py:56`）；无 `torch.manual_seed`、`np.random.seed`、CUDA 种子、确定性算法标志。

**修复方式**：不可修（论文数字已定）；可自行设置：加 `torch.manual_seed(seed)`、`np.random.seed(seed)` 与 CUDA 种子。

**影响**：设置后"你自己的实验"可复现，但**数字不会是论文的数字**（论文数字来自未播种的随机序列）；且固定种子可能固化单次运行的偶然性（代表性下降）。

### C2. "3 random seeds" 与代码矛盾

**问题**：`trainLPR.py:256`：`seed=config.runs + 1`，3 次 run **共用同一数据种子**，只有未播种的 torch 在变。论文声称 "3 random seeds in each run" 无对应实现。

**修复方式**：不可修（论文表述）；可自行设置：seed 随 run_id 变化（如 `seed=run_id`）。

**影响**：3 次 run 数据不同 → std 更真实，但口径与论文报告的 std 不同（论文可能用了别的种子方案）。

### C3. 精确数值（含 ±std）无法复现

**问题**：Table 2 每个 `mean ± std` 锚定在作者未公开的种子三元组上。

**修复方式**：不可修，且不可自行设置：无种子三元组 + torch 未播种 ⇒ 逐位复现数学上不可能。可复现上限：机制 + 量级接近 + 相对排序。

**影响**：无修复路径；接受"量级接近"为上限。

### C4. 论文 Table 1 统计 = 原始未过滤值

**问题**：实测：原始 `skill_builder_data.csv` 4,217 用户/123 技能/525,534 行；按论文描述的过滤后为 3,695/122/458,092。论文报告 4.2k/123/525.5k = **原始值**（avg 124.6 = 525534/4217）。

**修复方式**：不可修（论文已出版）；可自行设置：用仓库原始数据重跑过滤统计。

**影响**：重算得到真实过滤后统计；继续引用论文数字则继承其"原始未过滤"误差。

### C5. Junyi 数据缺失

**问题**：`data/raw_data/` 无 junyi 原始文件（`junyi_ProblemLog_for_PSLC.txt`、`junyi_Exercise_table.csv`）。

**修复方式**：不可修（外部授权）；可自行解决：PSLC DataShop 账号下载（论文称 198.5k 学生/719 题/3936 万交互）。Assist09 原始数据已在仓库（82MB）。

**影响**：下载后需核对规模与预处理口径（过滤条件、session/时间列处理）；不同版本或过滤条件 → 数字偏差。

### C6. 环境依赖缺失

**问题**：验证环境初始无 torch/pandas/hydra/wandb/gensim 等任何依赖。

**修复方式**：可自行设置：安装 requirements.txt（torch 2.1.2+cu118 需 CUDA）+ GPU。算力要求：5000+1000 episode × 3 runs × 9 方法 × 3 模式 × 3 步长，单卡数天至数周。

**影响**：torch 2.1.2+cu118 是 2023 年老版本（当前已到 2.13）；用更新的 torch/CUDA 算子结果可能不同（非确定性、数值差异）；GPU 与 CPU 也有差异。

## D. 代码卫生

### D1. PolicyNet 调试残留

**问题**：`Agent/agent_utils.py`：`if torch.isnan(self.fc1.weight).any(): print(...); t=t`（空操作）；`temperature` 参数传入但构造里写死 `self.temperature = 1`。

**修复方式**：删空操作、temperature 改用参数。

**影响**：temperature 从写死 1 改为用参数会改变 softmax 锐度 → 动作采样分布 → 训练行为；AC/PPO 各 yaml 的 temperature 值不同（0.1~1），修复后这些配置的行为会变，基线数字可能对不上论文。

### D2. EMAValue 名不副实

**问题**：`utils.py`：名为 EMA 实为普通运行均值（`alpha=0.9` 从未使用）。

**修复方式**：改成真 EMA 或改名。

**影响**：只影响日志展示（train/reward 均值曲线），不影响训练与最终数值。

### D3. `Student.initial_logs` 字段错误

**问题**：`Envs/StudentDataload.py`：`initial_logs = {'item_ids': initial_corrects, 'corrects': initial_corrects}`（item_ids 误填 corrects）。

**修复方式**：修正字段（未被使用，无害）。

**影响**：无行为变化；修复是防患于未来使用。

### D4. env 加载 checkpoint 吞异常

**问题**：`Env_junyiDKT.py`：`except Exception as e: t=t`，加载失败静默继续用随机权重；配合 A3（`xxx.pth` 占位）构成"无感知假训练"风险链。

**修复方式**：except 里显式报错。

**影响**：checkpoint 缺失/不匹配直接中断训练（不再静默空转）；要求先备好 checkpoint，改变"开箱即跑"体验但这是正确行为。

## E. 整体评估

### E1. 坏快照

**问题**：单作者 4 天（2025-07-31 ~ 08-03）批量上传；发布后 10 个 commit 中 6 个纯注释公式编号重排、2 个重命名（漏 genRec.py）、1 个 yaml、1 个 README 补表，零功能修复、零运行痕迹。时间线：07-31 代码可导入 → 08-02 重命名后 import 即崩 → 08-03 仍在加 README 表格。**发布前"论文对齐清理"改坏了代码，作者从未重验**。

**修复方式**：不可修（历史事实）；代码缺陷可自行修复（见 A 块）。

**影响**：无修复路径，接受现状即可。

### E2. 论文/代码性质判定

论文真实（AAAI-26 camera-ready，页码/版权/基金号/代码链接齐全）；公式 Eq(5)(7)(8)(14)-(19)(16)(17) 与代码逐一对应（方法为真）；数字方差模式真实、DKT 指标与公开文献一致（非造假）。综合结论：**真实实验 + 粗糙论文 + 坏代码快照**；Assist09 单线修完阻塞后具备"机制跑通、量级接近"的现实可行性，逐位复现不现实。无需修复。
