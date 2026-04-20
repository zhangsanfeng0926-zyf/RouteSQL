# RouteSQL

本项目基于原始的 **DAIL-SQL** 代码库构建，并逐步演进为一个具备**路径感知（Path-aware）**、**子空间感知（Subspace-aware）**、**多候选（Multi-candidate）**能力的 Text-to-SQL 系统。系统集成了结构化修复（Structured Repair）和最终阶段仲裁（Final-stage Arbitration）机制。目前代码主要针对 Spider 及其变体数据集进行评估，利用 `dataset_min` 进行快速迭代，并使用 `spider` 全量数据进行最终验证。

## 项目定位

本项目并非对原始 DAIL-SQL 论文的简单重新实现，而是一个持续性的工程化与研究分支。我们的目标是通过以下技术手段提升基于大语言模型（LLM）的 Text-to-SQL 性能：

- **模式与值链接 (Schema and Value Linking)**
- **连接路径发现与排序 (Join-path Discovery and Ranking)**
- **模式/路径子空间构建 (Schema/Path Subspace Construction)**
- **基于结构化子空间提示的问题重写 (Question Rewriting under Structured Subspace Hints)**
- **两阶段 SQL 框架生成与填充 (Two-stage SQL Framework Generation and Filling)**
- **多路径候选生成 (Multi-route Candidate Generation)**
- **结构化修复 (Structured Repair)**
- **最终合并与重排序 (Final Merge and Reranking)**

## 演进历史

### Stage 0: 原始 DAIL-SQL 基线
项目起步于原始 DAIL-SQL 的提示工程（Prompt Engineering）流水线：
- 少样本示例选择 (Few-shot example selection)
- 提示词生成 (Prompt generation)
- 单次生成或自一致性（Self-consistency）SQL 生成

### Stage 1: 在 `DAIL-SQL` 仓库中的首次演进
在旧的本地仓库中完成了第一条主要的扩展线：
- **V1**: 引入 `stage1 + stage2` 模式，实现框架生成、填充与修复。
- **V2**: 引入 `stage1/stage2` 合并机制。
- **V3**: 将硬性的框架脚手架弱化为软约束，并增加 `schema-first` 和 `direct` 路由。
- **V4**: 增加执行结果统计、框架置信度、候选追踪以及更强的合并策略。

*注：该线路目前仅用于历史参考和对比。*

### Stage 2: `DAIL-SQL2` 中的方法级重构
本项目（当前仓库）包含了第二条主要的演进线：
- **V1**: 路径候选、路径评分、路径感知修复。
- **V2**: 路径感知路由选择及类型化修复。
- **V3**: 多路由支持奖励及更精细的类型化修复。
- **V4**: 模式/路径子空间、图一致性、结构化修复、路径调节的路由选择。
- **V5**: 增加子空间提示的问题重写、路径图子空间构建、图一致性评分、连接路径图修复、成对候选偏好（Pairwise preference）及面向 EM 的归一化。
- **V6 (当前活跃分支)**: 在 V5 的基础上，持续优化修复、合并与验证层（Verifier layer）。

## 当前仓库状态

- **当前版本标记 (`VERSION.txt`)**: `V6_qaware_simplemerge`
- **实际意义**: 
    - `V5` 是稳定的生成骨架（Backbone）。
    - `V6` 是在 V5 基础上的迭代，重点在于最终决策层而非全盘重构。
    - 全量 Spider 性能表现仍是目前的优化重点。

## 目录结构

- `scripts/python_tools/`: 核心流水线逻辑，包括 SQL 生成、修复、重排序和合并。
- `utils/schema_path_utils.py`: 模式图构建、连接路径枚举、路径评分及图一致性辅助工具。
- `utils/linking_process.py`: 链接预处理与模式扩展工具。
- `utils/linking_utils/`: 模式链接（Schema linking）与值链接（Value linking）的支持代码。
- `prompt/`: 提示词模板与示例选择器逻辑。
- `scripts/server/`: 用于 `dataset-min` 和全量 `Spider` 运行的可复用服务器端脚本。
- `results/`: 建议仅保留 `latest` 和已存档的 `runs/` 结果，保持仓库整洁。

## 核心模块

1. **链接层 (Linking Layer)**: 负责将自然语言问题映射到候选表、列和具体值。
2. **模式/路径推理层 (Schema / Path Reasoning Layer)**: 负责构建模式图、枚举可能的连接路径，并推导出模式/路径子空间。
3. **LLM 控制层 (LLM Control Layer)**: 负责 SQL 归一化、框架生成与解析、问题重写、路由规划、候选生成、结构化修复、候选排序及验证器诊断。
4. **最终合并层 (Final Merge Layer)**: 负责 Stage 1 和 Stage 2 的仲裁以及最终输出的选择。
5. **运行/流水线层 (Runtime / Pipeline Layer)**: 负责可重复的命令行执行以及运行记录管理。

## 忽略追踪说明 (.gitignore)

为了保持公共仓库的整洁，以下大型本地产物**不应**提交：
- 本地 Python 环境 (`venv/`, `.conda/`)
- 模型与 Embedding 缓存 (`.cache/`, `vector_cache/`)
- `dataset/` 下的本地数据集
- `logs/` 下的日志文件
- `results/` 下的实验输出（文档和总结除外）
- 私密环境变量文件 (`scripts/server/server.env`)
- 下载的第三方运行时依赖 (如 Stanford CoreNLP jar 包)

## 环境配置

在运行实验前，请确保本地已准备好以下资源：
- 在 `dataset/` 目录下放置 Spider 数据集。
- 在 `third_party/` 目录下安装 Stanford CoreNLP。
- 根据 `requirements.txt` 配置本地环境。
- 参考 `scripts/server/server.env.example` 创建 `scripts/server/server.env`。

## 近期目标

短期目标并非一次性增加大量启发式规则，而是稳定生成、修复和合并的完整技术栈，以确保系统的稳健性与可扩展性。
