# UNO-LPR 问题清单

基于 2026-08-31 的代码审计、实测运行与论文（UNO_AAAI2026.pdf，AAAI-26）交叉验证。每个块内独立编号。状态口径：**可修**（含修复方式）/ **不可修**（不可修时说明能否自行设置）。

## A. 运行阻塞（代码无法启动/空转）

### A1. 悬空导入 `POMDPDKT`

`Agent/genRec.py:6`。`POMDPDKT → UniLPR` 重命名遗漏，类从未定义。`import Agent` 即抛 `ImportError`，所有 agent 训练入口全挂。可修（已修）：删除该导入。

### A2. DKT 预训练保存必挂

`LifeLongKT/trainKT.py`。`os.makedirs(os.path.dirname(model_save_path))` 只建 `saved_models/DKT`，缺 `junyi/` 末层，`torch.save` 抛 `Parent directory ... does not exist`。实测复现。可修：改 `os.makedirs(model_save_path, exist_ok=True)`，一行。

### A3. `xxx.pth` 占位符 + 静默随机 DKT

`trainLPR.py setup_environment` 硬编码 `model_path = './LifeLongKT/saved_models/DKT/junyi/xxx.pth'`（字面量，初始 commit 就有）。trainKT 实际产出 `<随机数>_seed_42_best.pth`，文件名不匹配；文件缺失时静默回退随机 DKT → reward 恒 0、actor_loss 恒 0，训练完全空转（实测）。可修：glob 最新 checkpoint + 缺失时 `raise FileNotFoundError`。

### A4. 默认入口损坏

`trainLPR.py` hydra `config_name="Random"`，但 `initialize_agent` 无 Random 分支 → AttributeError。可修：hydra 默认 config 改 `UNO`。

### A5. GRPO.yaml 死配置

`config/{junyi,assist09}/GRPO.yaml` 存在但从未有 GRPO agent（GRPO 目标内联在 `UNO.py:learn()`）。可修：删除或补实现。

## B. 论文 ↔ 代码协议不一致

### B1. 阈值语义不一致

env（`Env_junyiDKT.py`）用 DKT `states=True` 的**原始 logit** 判掌握/答对：`logit > 0.5`（⟺ `sigmoid > 0.622`）；trainKT 评估 ACC 用 `sigmoid(logit) > 0.5`（⟺ `logit > 0`）。env 阈值更严，奖励分布系统性偏移。可修：env 判掌握改 `sigmoid(x) > 0.5`（一行），阈值可自行设置。注意改后奖励分布会变，无法确定哪个口径对应论文数字。

### B2. UniLPR 预训练缺失

论文："Before generating recommendations, we train the model using λ₁L_KL + λ₂L_KS"（Eq 16/17 双监督预热）。代码：`UNO.__init__` 随机初始化 `UniLPR(...)`，无任何 load_state_dict；`begin_episode` 仅每 episode 做一个监督步，预训练产物与主流程脱节。可修：trainKT 产出 checkpoint 后让 UNO 加载；预训练轮数可自行设置，但论文未给细节。

### B3. EM/RM 双模型缺失

论文：环境模型 EM + 奖励模型 RM 两个预训练 DKT，RM 测试时停用。代码：只加载**一个** DKT，模拟与状态同源，无停用逻辑（`outside_model = None` 注释暗示作者曾计划/使用第二模型）。可修：改 env 加载第二个 DKT + 测试停用逻辑；或自行选择接受单模型近似。

### B4. 数据划分不一致

论文：85/15 训练/测试；代码 `trainKT.py`：80/10/10（README 同）。可修：改 `train_ratio`。

### B5. KT 训练 epoch 不一致

README：100 epochs；`DKT.yaml`：`num_epochs: 50`。论文未给具体值。可修：改 yaml，epoch 可自行设置。

### B6. Case study 死配置

`UNO.yaml` 有 `run_case_study: true`，但 `trainLPR.py` 从未读取；论文 Figure 5（学生 29144 案例）不可从仓库复现。可修（需实现，工作量中等）；不可自行设置。

### B7. rct 目标数表述不符

README："只有 1–5 target nodes"；代码 `rct-5/10/20` 设 `num_goals: 500`（作为上限，实际=去重后目标数）。可修：改配置；轻微。

### B8. README 的 DKT ACC/AUC 表论文没有

README 报 Junyi 0.8217/0.8475、Assist09 0.7328/0.7703；论文正文（含附录提取文本）无此表。不可修（论文/README 表述问题，非代码）；可自行用仓库数据重算核对。

### B9. README 表格 trian/final 行论文没有

论文 Table 2 只报测试（eval）值；README 额外加训练阶段 trian/final 行（来源应为训练日志）。不可修（表述差异）；可自行从训练日志复现。

## C. 复现性障碍

### C1. 种子未固化

全仓库仅一行 `random.seed(seed)`（`StudentDataload.py:56`）；无 `torch.manual_seed`、`np.random.seed`、CUDA 种子、确定性算法标志。不可修（论文数字已定），可自行设置：加 `torch.manual_seed(seed)`、`np.random.seed(seed)` 与 CUDA 种子即可获得确定性。

### C2. "3 random seeds" 与代码矛盾

`trainLPR.py:256`：`seed=config.runs + 1`，3 次 run **共用同一数据种子**，只有未播种的 torch 在变。论文声称 "3 random seeds in each run" 无对应实现。不可修（论文表述），可自行设置：把 `seed=config.runs + 1` 改为随 run_id 变化（如 `seed=run_id`）。

### C3. 精确数值（含 ±std）无法复现

Table 2 每个 `mean ± std` 锚定在作者未公开的种子三元组上。不可修，且不可自行设置：无种子三元组 + torch 未播种 ⇒ 逐位复现数学上不可能。可复现上限：机制 + 量级接近 + 相对排序。

### C4. 论文 Table 1 统计 = 原始未过滤值

实测：原始 `skill_builder_data.csv` 4,217 用户/123 技能/525,534 行；按论文描述的过滤后为 3,695/122/458,092。论文报告 4.2k/123/525.5k = **原始值**（avg 124.6 = 525534/4217）。不可修（论文已出版），可自行设置：用仓库原始数据重跑过滤统计。

### C5. Junyi 数据缺失

`data/raw_data/` 无 junyi 原始文件（`junyi_ProblemLog_for_PSLC.txt`、`junyi_Exercise_table.csv`）。不可修（外部授权），可自行解决：PSLC DataShop 账号下载（论文称 198.5k 学生/719 题/3936 万交互）。Assist09 原始数据已在仓库（82MB）。

### C6. 环境依赖缺失

验证环境初始无 torch/pandas/hydra/wandb/gensim 等任何依赖。可自行设置：安装 requirements.txt（torch 2.1.2+cu118 需 CUDA）+ GPU。算力要求：5000+1000 episode × 3 runs × 9 方法 × 3 模式 × 3 步长，单卡数天至数周。

## D. 代码卫生

### D1. PolicyNet 调试残留

`Agent/agent_utils.py`：`if torch.isnan(self.fc1.weight).any(): print(...); t=t`（空操作）；`temperature` 参数传入但构造里写死 `self.temperature = 1`。可修：删空操作、temperature 改用参数。

### D2. EMAValue 名不副实

`utils.py`：名为 EMA 实为普通运行均值（`alpha=0.9` 从未使用）。可修：改成真 EMA 或改名。

### D3. `Student.initial_logs` 字段错误

`Envs/StudentDataload.py`：`initial_logs = {'item_ids': initial_corrects, 'corrects': initial_corrects}`（item_ids 误填 corrects）。可修（未被使用，无害）。

### D4. env 加载 checkpoint 吞异常

`Env_junyiDKT.py`：`except Exception as e: t=t`，加载失败静默继续用随机权重；配合 A3（`xxx.pth` 占位）构成"无感知假训练"风险链。可修：except 里显式报错。

## E. 整体评估

### E1. 坏快照

单作者 4 天（2025-07-31 ~ 08-03）批量上传；发布后 10 个 commit 中 6 个纯注释公式编号重排、2 个重命名（漏 genRec.py）、1 个 yaml、1 个 README 补表，零功能修复、零运行痕迹。时间线：07-31 代码可导入 → 08-02 重命名后 import 即崩 → 08-03 仍在加 README 表格。**发布前"论文对齐清理"改坏了代码，作者从未重验**。不可修（历史事实）；代码缺陷可自行修复（见 A 块）。

### E2. 论文/代码性质判定

论文真实（AAAI-26 camera-ready，页码/版权/基金号/代码链接齐全）；公式 Eq(5)(7)(8)(14)-(19)(16)(17) 与代码逐一对应（方法为真）；数字方差模式真实、DKT 指标与公开文献一致（非造假）。综合结论：**真实实验 + 粗糙论文 + 坏代码快照**；Assist09 单线修完阻塞后具备"机制跑通、量级接近"的现实可行性，逐位复现不现实。评估结论，无需修复。
