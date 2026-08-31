# UNO-LPR 仓库验证与复现评估

验证时间：2026-08-31
验证方式：静态代码审计 + 实际运行复现（CPU 版 torch + 合成数据端到端冒烟）+ 论文（UNO_AAAI2026.pdf，AAAI-26 camera-ready）三方交叉验证；所有测试产物已清理，仓库保持干净。

## 一句话结论

**代码是"坏快照"：作者用本地代码跑出过真实实验，但发布前清理（POMDPDKT→UniLPR 重命名 + 注释对齐论文公式）改坏了代码且从未重新验证——当前代码按 README 无法开始训练。论文真实（AAAI-26 正式录用）、方法-代码数学一一对应、非造假；但论文本身粗糙（Table 1 统计抄的是未过滤的原始值）。完整复现论文全部数字不现实，但 Assist09 单条线（数据已在仓库）修完阻塞后具备"跑通机制、量级接近"的现实可行性。**

---

## 一、硬阻塞与修复方式（代码级）

### 1. import 崩溃 —— ✅ 已修复（2026-08-31）
- 位置：`Agent/genRec.py:6`，`POMDPDKT → UniLPR` 重命名遗漏，悬空导入
- 修复：删除 `POMDPDKT`，改 `from LifeLongKT import related_loss, SASRec_KT, GRU4Rec_KT`；连带修复 `LifeLongKT/trainKT.py:105-106` 模型分支（`elif config.model_name == 'UniLPR': model = UniLPR(...)`）、`configs/POMDPDKT.yaml` 重命名为 `UniLPR.yaml`
- 验证：`python -c "import Agent, trainLPR"` 无 stub 通过；UniLPR 预训练冒烟跑通

### 2. DKT 预训练保存必挂 —— ❌ 未修复，一行
- 位置：`LifeLongKT/trainKT.py` 保存逻辑
- 原因：`os.makedirs(os.path.dirname(model_save_path))` 只建了 `saved_models/DKT`，缺 `junyi/` 末层 → `torch.save` 报 `Parent directory ... does not exist`（实测复现）
- 修复方式：
```python
model_save_path = os.path.join(config.train.model_save_path, f"{config.model_name}/{config.dataset_name}")
os.makedirs(model_save_path, exist_ok=True)   # 替换原 dirname 逻辑
model_save_path = os.path.join(model_save_path, f"{num_exp}_seed_{config.seed}_best.pth")
```

### 3. `xxx.pth` 占位 + 静默随机 DKT —— ❌ 未修复，几行
- 位置：`trainLPR.py` `setup_environment()`，硬编码 `model_path = './LifeLongKT/saved_models/DKT/junyi/xxx.pth'`（字面量，从初始 commit 就在，从未填过）
- 风险：文件缺失时 `os.path.exists` 为假 → **静默回退随机 DKT，无警告** → 所有 reward 恒 0、actor_loss 恒 0，训练完全空转（实测）
- 修复方式：
```python
import glob
cands = sorted(glob.glob('./LifeLongKT/saved_models/DKT/junyi/*_best.pth'))
if not cands:
    raise FileNotFoundError('DKT checkpoint not found — run LifeLongKT/trainKT.py first')
model_path = cands[-1]   # 并删除静默回退：文件不存在必须报错
```
- 同时建议：env 加载失败路径（`except Exception as e: t=t`）改为显式报错，杜绝无感知回退

### 4. 默认入口损坏（Random/GRPO 死配置）—— ❌ 未修复
- `trainLPR.py` hydra 默认 `config_name="Random"`，但 `initialize_agent` 无 Random 分支 → 返回 None → AttributeError；`GRPO.yaml` 从未有 agent 实现（GRPO 目标内联在 `UNO.py:learn()`）
- 修复方式：hydra 默认改 `config_name="UNO"`；删除 `config/{junyi,assist09}/GRPO.yaml` 与 `Random.yaml`（或补实现）

### 5. 数据与依赖前置 —— 部分在仓库
- Assist09 原始数据**已在仓库**：`data/raw_data/assist09/skill_builder_data.csv`（82MB，525,534 行；此前误记为"空目录"，实测确认在）
- Junyi 原始数据缺失（`junyi_ProblemLog_for_PSLC.txt`、`junyi_Exercise_table.csv`），需 PSLC DataShop 账号下载
- `processed_data/` 由预处理脚本生成；本机无任何 Python 依赖（需 `requirements.txt`，torch 2.1.2+cu118 需 CUDA 环境）

---

## 二、完整复现的卡点与可行性

| 卡点 | 现状 | 能否解决 | 说明 |
|---|---|---|---|
| Assist09 数据 | 仓库已有 82MB 原始 CSV | ✅ 已解决 | 按代码过滤后 3,695 人 / 122 题 / 45.8 万交互 |
| Junyi 数据 | 缺失 | ⚠️ 可解决 | PSLC DataShop 账号下载；论文称 198.5k 学生 / 719 题 / 3936 万交互 |
| Python 依赖 | 本机无 | ✅ 可解决 | `requirements.txt`；torch 2.1.2+cu118 |
| GPU 算力 | 无 | ⚠️ 可解决 | 5000 训练 + 1000 测试 × 3 runs × 9 方法 × 3 模式 × 3 步长，单卡数天至数周 |
| 阻塞 #2/#3/#4 | 见第一节 | ✅ 简单修改 | 总计约 10 行代码 |
| EM/RM 双模型 | 论文：EM+RM 两个预训练 DKT、测试时关闭 RM；代码：只加载**一个** DKT | ⚠️ 部分 | 需改 env 加载第二模型 + 关闭逻辑；或接受单模型近似（奖励信号来源仍是 DKT 内部状态） |
| 阈值语义 | KT 训练 ACC 用 `sigmoid(logit)>0.5`；env 判掌握用**原始 logit>0.5**（≈sigmoid>0.62） | ✅ 可解决 | env 改 `sigmoid(x)>0.5` 或保持——影响奖励分布，不改也能跑 |
| torch 种子 | 未固定（仅数据采样种子固定） | ✅ 可解决 | main 里 `torch.manual_seed` + 确定性算法；±std 复现仍依赖环境 |
| KT 训练 epoch | 代码配置 50，README 说 100 | ✅ 可解决 | 改 yaml；论文未给具体值 |
| 数据划分 | 代码 80/10/10，论文写 85/15 | ✅ 可解决 | 改 `trainKT.py` train_ratio |
| Case study（论文 Figure 5） | `run_case_study: true` 死配置，`trainLPR.py` 从未读取 | ⚠️ 需实现 | 论文案例图不可从仓库复现 |
| 论文 Table 1 统计 | 报告值 = 原始文件值（见第三节） | — | 过滤后实际 3,695 人 / 45.8 万交互，与报告 4.2k / 525.5k 不符 |

**可行性结论**：修完阻塞 #2/#3/#4 + 装依赖 + GPU 后，**Assist09 单线（预处理 → DKT 预训练 → UNO 训练）具备完整可运行性**，目标定为"机制跑通、量级接近"，不保证逐位复现表格（无种子、协议漂移、EM/RM 缺失都会引入偏差）。Junyi 线补数据后可跑但无验证锚点。

---

## 三、论文/代码评估

### 论文真实性
- `UNO_AAAI2026.pdf` 为 **AAAI-26（第 40 届 AAAI）正式 camera-ready**：页码 15617–15625、AAAI 版权行、NSFC 基金号（U2469205）、代码链接 `github.com/PengLinzhi/UNO-LPR` 指向本仓库
- 时间线自洽：仓库上传于 2025-07-31（AAAI-26 截稿约 2025-07-25 之后），论文公式编号与代码注释编号逐一对应（`# (16) KL(kt_loss)` 等）——代码注释就是对着这篇论文写的

### 论文 ↔ 代码数学一致性（方法为真的核心证据）
| 论文 | 代码 |
|---|---|
| Eq(5) 过程奖励 = Ep 逐差 | `UNO.py learn()`：`advantages[1:] -= advantages[:-1]`（env reward 是累计 Ep，逐差还原 $g_i$）✓ |
| Eq(7) PAdv 学生序列内标准化 | `(advantages - mean)/(std + 1e-8)`，memory 每 episode 重置 ✓ |
| Eq(8) 统一编码 + `[REC]` token | `UniLPR.forward`：goal 均值嵌入 + action=2 的 `[REC]` 位 ✓ |
| Eq(14)(15) 温度 softmax 采样 | `logits/temperature → softmax → Categorical.sample` ✓ |
| Eq(16) BCE + Eq(17) NCE 双监督 | `kt_loss` + `related_loss`（负采样对比）✓ |
| Eq(18)(19) GRPO 组相对裁剪 | `-min(ratio·adv, clamp(ratio, 1±ε)·adv)` ✓ |
| rct/his/exp 三模式 | 代码 `rct`（前 60% 历史+后 20% 目标）/`revise`（历史随机 10）/`learn`（未见题随机 10）✓ |
| 5000/1000 episodes × 3 runs | `max_episodes: 5000`、`evaluate(1000)`、`runs: 3` ✓ |
| Table 2 测试数值 | 与 README eval 行逐字一致 ✓ |

### 可验证的统计出入（实测）
论文 Table 1 报告 Assist09 **4.2k 学生 / 123 题 / 525.5k 交互 / 平均 124.6**，并声称"过滤掉作答<5 次的题目与<5 次的学生"。对仓库内原始文件实测：
```
原始文件:      4,217 用户 / 123 技能 / 525,534 行
按代码过滤后:  3,695 用户 / 122 技能 / 458,093 行
论文报告:      4.2k   / 123   / 525.5k  ← 就是原始值（avg 124.6 = 525534/4217）
```
论文描述（过滤）与其 Table 1（原始统计）脱节——写论文时未重跑统计，属粗糙非造假。

### 坏快照判定（commit 历史证据链）
- 单作者（初始 `plz <626572397@qq.com>`，后续 `PengLinzhi`），16 个原始 commit 集中在 2025-07-31 ~ 08-03 四天，初始 commit 一次性包含全部代码 + 82MB 数据集 + `result.pdf`
- 发布后 10 个 commit：**6 个纯注释公式编号重排**（对齐论文草稿）、2 个 POMDPDKT→UniLPR 重命名（漏 `genRec.py`）、1 个 yaml 改名、1 个 README 补结果表——**零功能修复、零运行痕迹**（无 checkpoint/log/outputs 提交）
- 时间线：07-31 代码可导入 → 08-02 重命名后 import 即崩 → **08-03 作者仍在加 README 表格**（此时代码已不可运行）→ 从未重新验证
- `xxx.pth` 占位从初始 commit 就在；README 表格（08-03）晚于代码损坏点（08-02）
- 佐证"真实实验"的反向证据：数字方差模式真实（GRU4Rec/SASRec 部分 cell 爆到 ±17~±35 而 UNO/AC/PPO 稳定，3 次运行发散特征难以编造）；DKT 的 ACC/AUC（0.82/0.85、0.73/0.77）与领域公开文献一致；连 82MB 原始数据集都提交了

### 综合评估
**真实实验 + 粗糙论文 + 坏代码快照**，非造假：
1. 论文真实、方法真实（公式-代码一一对应，特化设计如 PAdv 逐差、`[REC]` token、组相对裁剪不可能凭空编造）
2. 作者确实跑过实验（数字自洽、方差真实、数据集统计与原始文件吻合）
3. 发布的代码是"论文对齐清理"时改坏的快照，作者从未重验发布版能否运行
4. 论文写作粗糙：Table 1 抄原始统计、85/15 与代码 80/10/10 不符、EM/RM 双模型代码未实现、case study 死配置、README 的 DKT ACC/AUC 表论文正文没有

---

## 四、已完成的修复（2026-08-31，commit 966e17f）

1. `Agent/genRec.py:6`：删除悬空 `POMDPDKT` 导入
2. `LifeLongKT/trainKT.py`：模型分支 `POMDPDKT` → `UniLPR`
3. `LifeLongKT/configs/POMDPDKT.yaml` → `git mv` 重命名 `UniLPR.yaml`
4. 新增 `.gitignore`（`data/raw_data/`、`data/processed_data/`、`LifeLongKT/saved_models/`、`outputs/`、`wandb/`、`__pycache__/`）
5. 新增本文档

验证：import 无 stub 通过；UniLPR 预训练冒烟（合成数据 1 epoch）跑通并保存 checkpoint。

## 五、修补后实测结果

打上 POMDPDKT stub + 合成数据 + 预创建保存目录 + checkpoint 改名后：3 个训练 episode + 1000 episode 评估完整跑通无异常（~90 秒 CPU），`seq_kt_loss` 正常下降，梯度流正常；但 reward 恒 0——因欠训练 DKT 的 logits 过不了 0.5 阈值，学习信号完全依赖真正训练好的 DKT。

## 六、风险提示

1. `xxx.pth` 静默回退风险最高：不修就复现，会产出全 0 的"假训练结果"而毫无察觉；任何复现前必须先确认 env 加载了训练好的 DKT
2. 论文 Table 1 与真实过滤后统计不一致（4.2k/525.5k vs 3,695/45.8 万）——引用论文数据时注意
3. EM/RM 双模型、阈值语义、种子等协议漂移会让数字与论文表格有系统性偏差，"量级接近"已是现实上限
