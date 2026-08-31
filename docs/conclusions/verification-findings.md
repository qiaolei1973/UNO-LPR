# UNO-LPR 仓库可运行性验证结论

验证时间：2026-08-31
验证方式：静态代码审计 + 实际运行复现（安装 CPU 版 torch 与最小依赖，构造合成数据做端到端冒烟测试；所有测试产物已清理，仓库保持干净）

## 一句话结论

**当前代码按 README 操作无法开始训练，更无法复现论文结果。** 存在 3 个可实测复现的硬阻塞（import 崩溃、模型保存路径 bug、checkpoint 占位符），另有数据与依赖前置条件缺失。修补阻塞后代码机制上可跑通完整训练+评估循环，但学习信号依赖真正训练好的 DKT，数值复现缺证据链。

## 修复更新（2026-08-31）

### 改变了什么
1. `Agent/genRec.py:6`：删除悬空导入 `POMDPDKT`（`POMDPDKT → UniLPR` 重命名遗漏），import 行现为 `from LifeLongKT import related_loss, SASRec_KT, GRU4Rec_KT`
2. `LifeLongKT/trainKT.py:105-106`：模型实例化分支由 `elif config.model_name == 'POMDPDKT': model = POMDPDKT(...)` 改为 `elif config.model_name == 'UniLPR': model = UniLPR(...)`
3. `LifeLongKT/configs/POMDPDKT.yaml` → `git mv` 重命名为 `LifeLongKT/configs/UniLPR.yaml`（内容 `model_name: UniLPR` 此前已改）
4. 新增 `.gitignore`：忽略训练产出（`data/raw_data/`、`data/processed_data/`、`LifeLongKT/saved_models/`、`outputs/`、`wandb/`、`__pycache__/` 等）
5. 新增本文档 `docs/conclusions/verification-findings.md`

### 改变后实测验证
- `python -c "import Agent, trainLPR"` **无 stub 直接通过**（修复前：ImportError）
- UniLPR 预训练分支冒烟：`python -m trainKT --config-name UniLPR --config-path configs`（合成数据、1 epoch、`num_negative=3`）正常训练、评估并保存 checkpoint `saved_models/UniLPR/junyi/<rand>_seed_42_best.pth`
- 测试产物已清理，仓库仅含上述代码改动

### 改变后的可运行性结论
- ✅ **阻塞 #1（import 崩溃）已修复**：所有 agent 的训练入口可正常导入
- ❌ **阻塞 #2（trainKT 保存目录缺层）仍存在**：`os.makedirs(os.path.dirname(model_save_path))` 少建末层目录（如 `junyi/`），`torch.save` 报 `Parent directory ... does not exist`；冒烟测试通过预创建目录绕过
- ❌ **阻塞 #3（`xxx.pth` 占位 + 静默随机 DKT 回退）仍存在**：`trainLPR.py setup_environment` 硬编码 `./LifeLongKT/saved_models/DKT/junyi/xxx.pth`，与 trainKT 实际产出文件名不一致；文件缺失时静默使用随机 DKT，训练 reward 恒为 0（空转）
- 数据与依赖前置条件不变：需下载数据集 → 预处理 → 安装依赖 → GPU

## 硬阻塞（按出现顺序，均已实测复现）

### 1. import 即崩——所有 agent 的训练入口全挂
- 位置：`Agent/genRec.py:6` `from LifeLongKT import POMDPDKT, ...`
- 事实：`POMDPDKT` 全仓库不存在（grep 确认仅在 `genRec.py:6` 与 `trainKT.py:105` 被引用，从未定义，`LifeLongKT/__init__.py` 也未导出）
- 实测：`python -m trainLPR` → `ImportError: cannot import name 'POMDPDKT' from 'LifeLongKT'`
- 影响：`trainLPR.py` 的 `from Agent import ...` 触发 `Agent/__init__.py` → 任何 `--config-name` 都启动不了

### 2. DKT 预训练保存必挂
- 位置：`LifeLongKT/trainKT.py` 保存逻辑 `os.makedirs(os.path.dirname(model_save_path))`
- 事实：`model_save_path = ./saved_models/DKT/junyi`，`dirname()` 只返回 `./saved_models/DKT`，`junyi/` 层未创建
- 实测：训练循环正常跑完，`torch.save` 抛 `RuntimeError: Parent directory ./saved_models/DKT/junyi does not exist`
- 手工预创建目录后保存成功，产出 `<随机数>_seed_<seed>_best.pth`

### 3. 环境 checkpoint 硬编码占位符 `xxx.pth`
- 位置：`trainLPR.py` `setup_environment()` 中 `model_path = './LifeLongKT/saved_models/DKT/junyi/xxx.pth'`
- 事实：与 trainKT 实际产出文件名（`<随机数>_seed_42_best.pth`）不一致；文件不存在时 `os.path.exists` 为假，**静默回退到随机初始化 DKT，无任何警告**
- 实测：随机 DKT 输出 logits 全落在 [-0.05, 0.05]，env 以 `logit > 0.5` 判掌握恒为 False → 所有 episode reward 恒为 0、actor_loss 恒为 0，训练完全空转
- 手动改名 `xxx.pth` 后加载成功（state_dict 逐参数匹配），但欠训练的 DKT（合成数据上 AUC 0.42–0.57）同样给不出非零 reward

### 4. 数据与依赖前置条件缺失
- `data/raw_data/` 只有 assist09 空目录；junyi 原始文件（`junyi_ProblemLog_for_PSLC.txt`、`junyi_Exercise_table.csv`）与 `processed_data/` 均不在仓库
- 预处理脚本需要先从 PSLC DataShop 下载 ASSISTments09 / Junyi 数据集
- 验证环境（Python 3.11.6）初始无 torch/pandas/hydra/wandb/gensim 等任何依赖

## 其他已确认的问题（非启动阻塞）

- **默认入口损坏**：hydra 默认 `config_name="Random"`，`Random.yaml` 设 `agent: Random`，但 `initialize_agent` 无 Random 分支 → 返回 None → 运行即 AttributeError
- **GRPO.yaml 死配置**：`config/{junyi,assist09}/GRPO.yaml` 存在，但无 GRPO agent 实现（GRPO 目标内联在 `UNO.py:learn()` 中，配置为遗留物）
- ~~**UniLPR 无法通过 trainKT.py 预训练**~~（已于 2026-08-31 修复）：原 `train_kt` 只有 DKT/POMDPDKT 分支；现改为 UniLPR 分支，`--config-name UniLPR --config-path configs` 可正常预训练（依赖 `config.train.num_negative`，仅 UniLPR.yaml 提供）
- **阈值语义不一致**：KT 训练期 ACC 用 `sigmoid(logit) > 0.5`，env 判掌握用**原始 logit > 0.5**（≈ sigmoid > 0.62），改变了奖励分布
- **确定性不足**：torch 种子未固定，仅数据采样种子固定；run 间差异靠权重随机初始化，±std 复现依赖完整环境
- 代码卫生：`PolicyNet` 残留调试空操作（`if torch.isnan(...): t=t`）；`EMAValue` 实为普通运行均值（alpha 未用）；`Student.initial_logs` 中 `item_ids` 误填 `initial_corrects`（未被使用，无害）

## 修补后的实测结果

打上 POMDPDKT stub + 合成数据 + 预创建保存目录 + checkpoint 改名后：
- **完整循环跑通无异常**：3 个训练 episode + 1000 episode 评估，约 90 秒完成（CPU）
- `seq_kt_loss` 正常下降（0.1454 → 0.1397），优化器与梯度流正常
- **但 reward 恒为 0**：因 DKT 欠训练（合成随机数据无技能结构），logits 过不了 0.5 阈值
- 结论：代码机制本身无更多隐藏崩溃；学习信号完全依赖一个真正训练好的 DKT（论文报 AUC 0.77–0.85）

## 复现论文结果的评估

即使修完所有阻塞，复现仍有以下障碍：
1. 需真实数据集（Junyi / ASSIST09，需 PSLC DataShop 账号下载）
2. 需 GPU：5000 训练 + 1000 评估 episode × 3 runs × 9 方法 × 3 模式 × 3 步长，单卡数天至数周
3. 阈值语义、UniLPR 预训练缺失等问题使实验协议与论文描述不完全一致
4. 论文表格的精确数值（含 ±std）无种子固化，无法端到端核对

## 最小修复路径

1. ✅ 已完成（2026-08-31）：`Agent/genRec.py:6` 删除 `POMDPDKT` 导入；`trainKT.py` 模型分支改为 `UniLPR`；`configs/POMDPDKT.yaml` 重命名为 `configs/UniLPR.yaml`——恢复 import 与 UniLPR 预训练
2. `LifeLongKT/trainKT.py`：保存处改为 `os.makedirs(model_save_path, exist_ok=True)`
3. `trainLPR.py` `setup_environment`：改为 glob 最新 checkpoint 或从配置读取，删除 `xxx.pth` 占位
4. 下载数据集 → `data/preprocess_junyi.py` → `LifeLongKT/trainKT.py` → `trainLPR.py --config-name UNO`（GPU 环境）

## 风险提示

`xxx.pth` 静默回退设计风险最高：用户在不了解的情况下会产出全 0 的"假训练结果"而毫无察觉。任何复现尝试前必须先确认 env 实际加载了训练好的 DKT。
