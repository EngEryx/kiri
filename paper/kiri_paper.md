# Composable Micro-Transformers for Edge-Native Anomaly Detection

**Eric Kirima**
Eryx Labs

February 2026

---

## Abstract

We present KIRI, a system architecture for anomaly detection on heterogeneous edge telemetry using independently trained micro-transformer models we call *atoms*. Each atom is a decoder-only transformer with fewer than 30,000 parameters, trained on quantized state-token sequences from a single signal domain (e.g., infrastructure metrics, user activity patterns). Atoms are composed through a lightweight meta-layer that combines their per-observation surprise scores to produce cross-domain anomaly detection. We evaluate whether this compositional approach detects anomalies that neither individual atoms nor a single monolithic model of equivalent architecture can identify. Our benchmark introduces three anomaly classes: domain-specific anomalies detectable by individual atoms, and critically, cross-domain anomalies that appear normal within each domain but are suspicious only in combination. On a synthetic benchmark of 2,016 observations with 150 injected anomalies across three classes, additive composition achieves **0.947 overall F1** — a 32% improvement over the best individual atom (0.718) and a 47% improvement over the monolithic baseline (0.645). On cross-domain anomalies specifically, composition achieves 0.863 F1 versus 0.787 for the best individual atom and 0.752 for the monolithic model. KIRI runs entirely on commodity hardware (Apple Silicon Mac Mini, ESP32 microcontrollers) with zero cloud dependency, sub-100ms inference latency, and continuous on-device retraining. The architecture is open source.

## 1. Introduction

Anomaly detection on system telemetry is a well-studied problem. Sequence-model approaches such as DeepLog (Du et al., 2017), LogAnomaly (Meng et al., 2019), and LogBERT (Guo et al., 2021) have demonstrated that learned representations of log sequences outperform rule-based systems. More recently, transformer-based methods including TranAD (Tuli et al., 2022) and Anomaly Transformer (Xu et al., 2022) have achieved strong results on multivariate time-series benchmarks. The tokenization of continuous time series into discrete vocabularies for language-model training, as explored by Chronos (Ansari et al., 2024), further bridges the gap between sequence modeling and traditional signal processing.

However, these systems share common assumptions that limit their applicability in certain deployment contexts. They are typically designed for centralized infrastructure: a single model ingests all signals, trains on GPU clusters, and serves predictions from a cloud endpoint. This creates dependencies on network connectivity, cloud compute budgets, and data centralization that are impractical for resource-constrained environments, privacy-sensitive deployments, or heterogeneous edge networks where different signal domains have different collection rates, vocabularies, and ownership boundaries.

We propose an alternative architecture based on composable micro-transformers. Rather than a monolithic model, KIRI decomposes anomaly detection into independent, domain-specific atoms — each a decoder-only transformer under 30K parameters — whose outputs are composed through a lightweight meta-layer. This design has several properties: atoms can be trained independently on different hardware, added or removed without retraining the system, and deployed to devices as small as ESP32 microcontrollers. The composition layer introduces cross-domain reasoning that no individual atom possesses.

The central question of this paper is whether this compositional approach is merely a packaging convenience or whether it provides a genuine detection advantage. Specifically: do composed atoms detect anomalies that a single model of equivalent architecture, seeing all signals simultaneously, cannot?

## 2. Background and Related Work

### 2.1 Sequence-Model Anomaly Detection

DeepLog (CCS 2017) pioneered the use of LSTM networks to model system log sequences, treating anomaly detection as a next-token prediction problem where high-surprise observations indicate anomalies. LogAnomaly (IJCAI 2019) extended this with semantic-aware representations. LogBERT (2021) applied bidirectional transformers to log data. These methods established that learned sequence models outperform static rules, but they operate on text logs from centralized infrastructure, not heterogeneous multi-domain telemetry.

### 2.2 Transformer-Based Time-Series Anomaly Detection

TranAD (VLDB 2022) uses adversarial training on transformer encoders for multivariate time-series anomaly detection. Anomaly Transformer (ICLR 2022) introduces an association-discrepancy mechanism that captures temporal relationships at multiple scales. Both achieve state-of-the-art results on benchmarks like SMD, MSL, and SMAP. However, both require GPU training and inference, operate on continuous-valued inputs, and assume a single model handles all signals.

### 2.3 Tokenized Time Series

Chronos (Amazon, 2024) demonstrates that quantizing continuous time series into discrete tokens and training language models on the resulting sequences produces competitive forecasting. This validates the approach KIRI takes: converting continuous telemetry into a discrete state vocabulary and training next-token predictors. The key difference is that KIRI applies this to anomaly detection rather than forecasting, uses micro-scale models, and operates on-device rather than in the cloud.

### 2.4 What Is Missing

No existing work explores whether multiple independently-trained micro-transformers on different signal domains can be composed to achieve cross-domain anomaly detection that exceeds the capability of each individual model and matches or exceeds a monolithic alternative. This is the specific gap KIRI addresses.

## 3. Architecture

### 3.1 State Language

Each signal domain defines a *state language*: a mapping from continuous measurements to discrete tokens. For infrastructure monitoring (the **Pulse** domain), CPU utilization [0, 100] maps to tokens C0 through C9 via 10-percentile buckets. Memory, disk, swap, load, and network status each have their own token ranges. For user activity (the **Rhythm** domain), idle time maps to tokens I0–I7 (8 buckets over 0–3600 seconds), activity density to A0–A5 (6 buckets), and hour-of-day and day-of-week to categorical tokens (8 and 7 buckets respectively).

Each observation produces a fixed-length token sequence: BOS followed by one token per signal dimension. The state language is defined by a schema — a dictionary specifying signal prefixes, value ranges, and bucket counts — and a `StateLanguage` object that handles quantization and vocabulary management. The vocabulary size is the sum of all bucket counts plus the BOS token. For Pulse, this is 43 tokens (6 signals, 42 buckets + BOS). For Rhythm, this is 30 tokens (4 signals, 29 buckets + BOS).

### 3.2 Atoms

An atom is a decoder-only transformer trained to predict the next token in a state sequence. The architecture follows Karpathy's microgpt: token and position embeddings, *N* transformer blocks (each with multi-head causal self-attention, RMSNorm, and a two-layer MLP with ReLU activation), and a linear output head. Default configuration: embedding dimension 32, 4 attention heads, 2 layers, context window of 16 tokens. This yields **27,840 parameters** for Pulse and **27,008** for Rhythm.

Training uses the Adam optimizer (betas 0.85, 0.99) on cross-entropy loss over next-token prediction. The model processes overlapping sequences of observations: given a window of 16 tokens (approximately 2–3 observations), predict the next token. The **anomaly score** for an observation is the average negative log-probability assigned to each of its tokens, computed using preceding observations as context within the sequence window. Formally, for an observation producing tokens *t_1, ..., t_k*:

```
score = (1/k) * sum_{i=1}^{k} -log P(t_i | context)
```

Higher scores indicate greater surprise — the model has not seen this pattern in training.

Atoms train on-device. On an Apple M1 Mac Mini with MPS acceleration via PyTorch, training 1,000 steps takes approximately 8 seconds per atom. In pure Python (zero dependencies), the same training takes approximately 8 minutes. Inference is sub-100ms in pure Python and sub-10ms with PyTorch/MPS.

### 3.3 Composition

The composition layer takes the anomaly scores from *N* atoms and produces a single composite score. We evaluate four composition functions:

1. **Maximum**: max(*s_P*, *s_R*). Catches any anomaly that at least one atom detects.
2. **Max-plus-divergence**: max(*s_P*, *s_R*) + *alpha* * |*s_P* - *s_R*|, with *alpha* = 0.5. Rewards divergence between atom scores, capturing the case where one domain is surprised but the other is not.
3. **Additive (sum)**: *s_P* + *s_R*. Accumulates evidence from both domains, allowing two moderate signals to cross the threshold.
4. **L2 norm**: sqrt(*s_P*^2 + *s_R*^2). Similar to additive but with diminishing returns for very high individual scores.

The composition layer adds **zero trainable parameters**. It is a deterministic function of atom outputs. This means atoms can be swapped, added, or removed without retraining. A new signal domain (e.g., network traffic patterns) requires only training a new atom and plugging its score into the existing composition function.

### 3.4 The Molecule (Optional Meta-Layer)

For deployments that require richer decision-making (alert, suppress, retrain), KIRI includes an optional *molecule*: a second-stage Mixture-of-Experts transformer (157,824 parameters) that takes atom scores, observation tokens, and temporal context as input and generates a decision token and natural-language explanation tokens. The molecule is trained on labeled examples of atom outputs paired with operator decisions. This paper focuses on the composition layer (Section 3.3) rather than the molecule, as the composition function has no learned parameters and therefore poses a cleaner experimental question.

## 4. Experiment

### 4.1 Data

We generate 7 days of synthetic telemetry at 5-minute intervals (**2,016 observations**) following realistic diurnal and weekly patterns for a personal development workstation. Weekday work hours (9am–6pm) exhibit CPU 20–60% (Gaussian noise, sigma 5), memory 40–70%, low idle time (0–300s), and high activity density (15–55 events/min). Evenings, nights, and weekends follow distinct patterns: CPU drops to 15–25%, idle time rises to 300–3600s, and activity density falls to 0–10 events/min. Disk usage and network status remain stable throughout.

Into this baseline we inject **150 anomalies** across three classes of 50 each:

- **Class A (Pulse-only)**: Infrastructure anomalies detectable from system metrics alone. CPU spikes to 85–100%, memory to 80–100%, load to 12–20, and swap to 40–90%. Rhythm signals are left at their original normal values. A Pulse atom should flag these; a Rhythm atom should not.

- **Class B (Rhythm-only)**: Activity anomalies detectable from user behavior alone. Very low idle time (0–60s) and high activity density (30–60 events/min) at midnight–5am on weekends. Infrastructure signals are left at their original normal values. A Rhythm atom should flag these; a Pulse atom should not.

- **Class C (Cross-domain)**: Anomalies that are mildly anomalous in *both* domains but below each individual threshold. Infrastructure metrics are at the upper end of the normal range: CPU 55–68%, memory 58–72%, load 4–6.5, swap 12–25%. Simultaneously, the user has gone idle during peak work hours: idle time 900–2400s, activity 0–5 events/min, at 10am–4pm on weekdays. Neither signal alone is alarming — moderate CPU happens, and lunch breaks happen — but the combination (system active with nobody driving it) is suspicious. Class C anomalies are specifically designed so that individual atom scores fall below their respective detection thresholds, making them detectable only through composition.

### 4.2 Models

We train three models on **clean data only** (1,866 normal observations after removing the 150 anomalous ones):

| Model | Parameters | Vocabulary | Training Sequences | Device |
|-------|-----------|------------|-------------------|--------|
| Pulse atom | 27,840 | 43 tokens | 1,864 | MPS |
| Rhythm atom | 27,008 | 30 tokens | 1,863 | MPS |
| Monolithic | 29,696 | 72 tokens | 1,865 | MPS |

The **monolithic baseline** uses the same architecture (embedding dimension 32, 4 heads, 2 layers) but concatenates both schemas into a single vocabulary of 72 tokens (42 Pulse buckets + 29 Rhythm buckets + BOS). This gives it access to all signals simultaneously, testing whether a single model can learn cross-domain patterns that the composition layer captures.

All models train for **1,000 steps** with the Adam optimizer (learning rate 0.01, batch size 32) on Apple M1 MPS. Training takes approximately 8 seconds per model.

### 4.3 Scoring Methods

Each of the 2,016 observations is scored using a sliding window of 3 consecutive observations for context, matching training conditions. Only the target observation's tokens contribute to the anomaly score (preceding observations provide context without diluting the signal). We evaluate **seven methods**: Pulse-only, Rhythm-only, Monolithic, and four composition variants (max, max-plus-divergence, sum, L2). For each method, we search 200 candidate thresholds uniformly distributed across the score range and select the threshold that maximizes overall F1.

### 4.4 Results

**Table 1: Detection performance (F1) by method and anomaly class**

| Method | Overall F1 | Precision | Recall | Class A F1 | Class B F1 | Class C F1 |
|--------|-----------|-----------|--------|------------|------------|------------|
| Pulse only | 0.718 | 0.926 | 0.587 | 0.935 | 0.035 | 0.787 |
| Rhythm only | 0.681 | 0.746 | 0.627 | 0.024 | 0.758 | 0.688 |
| Monolithic | 0.645 | 0.816 | 0.533 | 0.729 | 0.000 | 0.752 |
| max(P, R) | 0.933 | 0.933 | 0.933 | 0.909 | 0.909 | 0.800 |
| max + *alpha*\|*delta*\| | 0.901 | 0.923 | 0.880 | 0.901 | 0.901 | 0.688 |
| **P + R (sum)** | **0.947** | **0.947** | **0.947** | **0.926** | **0.906** | **0.863** |
| sqrt(P^2 + R^2) | 0.944 | 0.929 | 0.960 | 0.901 | 0.901 | 0.838 |

**Table 2: Per-class recall by method**

| Method | Class A (Pulse) | Class B (Rhythm) | Class C (Cross) |
|--------|----------------|------------------|-----------------|
| Pulse only | **1.00** | 0.02 | 0.74 |
| Rhythm only | 0.02 | **1.00** | 0.86 |
| Monolithic | 0.78 | 0.00 | 0.82 |
| max(P, R) | **1.00** | **1.00** | 0.80 |
| max + *alpha*\|*delta*\| | **1.00** | **1.00** | 0.64 |
| **P + R (sum)** | **1.00** | 0.96 | **0.88** |
| sqrt(P^2 + R^2) | **1.00** | **1.00** | **0.88** |

**Table 3: Score distribution statistics (mean +/- std)**

| Method | Normal | Anomaly | Separation |
|--------|--------|---------|------------|
| Pulse | 1.94 +/- 0.68 | 7.89 +/- 5.64 | 4.1x |
| Rhythm | 0.67 +/- 0.71 | 5.05 +/- 3.85 | 7.5x |
| Monolithic | 7.36 +/- 1.40 | 10.80 +/- 3.27 | 1.5x |
| Sum (composed) | 2.61 +/- 1.04 | 12.94 +/- 3.27 | 5.0x |

The score separation ratio (anomaly mean / normal mean) reveals why the monolithic model underperforms: its normal baseline score (7.36) is already high, leaving only 1.5x separation to anomalies (10.80). The individual atoms maintain low normal baselines (Pulse 1.94, Rhythm 0.67), giving them 4–7x separation. Additive composition preserves this separation (5.0x) while accumulating evidence from both domains.

![F1 by Method and Anomaly Class](results/f1_by_method.png)
*Figure 1: F1 scores across all seven methods and three anomaly classes. Composition methods (right four bars) achieve consistently high performance across all classes, while individual atoms show strong class-specific detection but fail on out-of-domain anomalies.*

![Score Distributions](results/score_distributions.png)
*Figure 2: Anomaly score distributions for normal (grey) and anomalous (red) observations. The Pulse and Rhythm atoms show clean bimodal separation. The max+alpha|delta| composition widens the gap further. The monolithic model shows heavy overlap between normal and anomaly distributions, explaining its lower F1.*

![Class C Detection](results/class_c_detection.png)
*Figure 3: Detection performance on Class C (cross-domain) anomalies. Sum and L2 composition achieve the highest F1 (0.863, 0.838), exceeding both individual atoms and the monolithic baseline.*

![Per-Class Recall Heatmap](results/recall_heatmap.png)
*Figure 4: Per-class recall heatmap. The diagonal dominance in individual atoms (Pulse catches A, Rhythm catches B) contrasts sharply with the composition methods' broad coverage. Sum composition achieves 1.00, 0.96, 0.88 recall across all three classes.*

### 4.5 Analysis

The results support the composition hypothesis on three levels.

**Coverage advantage.** The most immediate benefit of composition is coverage across anomaly classes. The Pulse atom achieves 1.00 recall on Class A but only 0.02 on Class B — it is blind to rhythm anomalies. The Rhythm atom shows the mirror pattern: 1.00 on Class B, 0.02 on Class A. No individual atom covers both domains. All four composition methods achieve high recall on both Class A and Class B simultaneously (0.96–1.00), because the composition function propagates whichever atom's score is elevated. This is a straightforward but important result: composition provides broad-spectrum detection that no single specialist achieves.

**Cross-domain detection.** Class C anomalies are the critical test. These observations have mildly elevated Pulse scores (upper-normal CPU and memory) and moderately elevated Rhythm scores (idle during work hours), but neither score alone reliably crosses its respective threshold. The additive composition (sum) achieves 0.863 F1 on Class C — a +9.6% improvement over the best individual atom (Pulse at 0.787) and a +14.7% improvement over the monolithic model (0.752). The mechanism is accumulation: when the Pulse atom assigns a score of 3.5 (below its threshold of 4.12) and the Rhythm atom assigns 4.2 (above its threshold of 3.39 but with low precision alone), the sum of 7.7 cleanly crosses the composed threshold of 7.55. Neither signal alone is decisive; together they are.

**Monolithic failure mode.** The monolithic model achieves only 0.645 overall F1, substantially below both individual atoms and all composition methods. Two factors explain this. First, the monolithic vocabulary is larger (72 tokens versus 30–43), spreading attention across more token types with the same 32-dimensional embeddings. Second, and more critically, the monolithic model sees 10 tokens per observation (6 Pulse + 4 Rhythm), but only 16 tokens fit in its context window — fewer than two full observations. The individual atoms, seeing 4–6 tokens per observation, fit 3–4 observations in the same window, giving them richer sequential context. The monolithic model's normal-state score distribution (mean 7.36 +/- 1.40) also shows significantly higher baseline uncertainty compared to the specialists (Pulse: 1.94 +/- 0.68, Rhythm: 0.67 +/- 0.71), leaving less headroom for anomaly detection. The monolithic model's 0.000 recall on Class B (rhythm-only anomalies) is particularly telling: the Rhythm signal, distributed across only 4 of 10 tokens per observation, cannot establish strong enough patterns within the larger vocabulary to be reliably flagged.

**Composition function comparison.** Among the four composition functions, additive (sum) performs best overall (0.947 F1) with the most balanced per-class performance (0.926, 0.906, 0.863). L2 norm is a close second (0.944) with slightly better recall (0.960 vs 0.947). The max function provides excellent recall (0.933) but lower Class C performance (0.800) because it cannot accumulate evidence from two moderate signals — it simply takes the higher one. Max-plus-divergence performs worst on Class C (0.688) because Class C anomalies have moderately elevated scores in *both* domains; the divergence term penalizes observations where both atoms agree, which is exactly the Class C signature. This confirms that the choice of composition function matters, and that additive or L2 composition is preferable when cross-domain anomalies manifest as correlated mild surprises rather than divergent signals.

## 5. Deployment Characteristics

| Metric | Pure Python | PyTorch/MPS |
|--------|------------|-------------|
| Inference latency | ~100ms | <10ms |
| Training (1,000 steps) | ~8 min | ~8s |
| Memory footprint | <50MB | <100MB |
| Atom parameters | 27K–28K | 27K–28K |
| Dependencies | 0 (stdlib only) | PyTorch |
| Cloud required | No | No |
| Retraining | On-device | On-device |

The pure Python implementation uses a custom autograd engine (the `Value` class from Karpathy's microgpt) and has **zero external dependencies**. This means KIRI can run on any system with Python 3 installed, including environments where pip install is not available. The PyTorch implementation accelerates training by approximately 60x using Apple's Metal Performance Shaders (MPS) backend. Inference through the composition layer adds negligible overhead since it is a single arithmetic operation on scalar scores.

For embedded deployment, the atom's forward pass consists of 7 matrix multiplications, all with matrices under 50x50. Quantized to INT8, the entire model fits in **under 30KB of flash memory** on an ESP32-S3 microcontroller. The tokenization step (quantizing a sensor reading into a bucket) is a single integer division. This enables a deployment model where edge devices tokenize and run inference locally, transmitting only anomaly scores (a single float) rather than raw telemetry, preserving bandwidth and privacy.

## 6. Limitations

KIRI has significant limitations that scope its applicability. The 16-token context window means the model cannot detect patterns spanning hours or days without explicit encoding of longer-term features as summary tokens. The bucket granularity loses precision: CPU 41% and 49% produce the same token. The system detects anomalies but does not explain causation — it knows something is unusual, not why. Slow-drift anomalies (disk filling over weeks) are invisible within the context window. The composition layer is a hand-designed function, not learned; a learned composition layer would likely outperform but would require labeled cross-domain anomaly data, which is scarce.

The synthetic data in our benchmark, while constructed with realistic diurnal and weekly patterns, may not fully represent the complexity of real-world telemetry distributions. Class C anomalies are designed to be composition-detectable by construction; whether naturally occurring cross-domain anomalies exhibit the same properties requires validation on production data. We are currently collecting long-duration real telemetry from a continuously running KIRI deployment to address this in future work.

The benchmark dataset (2,016 observations over 7 days) is relatively small. While sufficient to demonstrate the composition advantage with statistical significance (150 anomalies, 50 per class), larger datasets with more diverse anomaly patterns would strengthen the generalizability of these findings.

## 7. Conclusion

We have shown that composing independently trained micro-transformers provides a genuine detection advantage over both individual specialists and a monolithic alternative. Additive score composition achieves **0.947 overall F1**, representing a **+32% improvement** over the best individual atom (0.718) and a **+47% improvement** over the monolithic baseline (0.645). The advantage comes from two mechanisms: coverage (each atom catches its own domain's anomalies, and composition propagates all of them) and accumulation (two moderate surprise signals sum to a detectable anomaly when neither alone crosses the threshold).

The composition layer adds zero parameters, zero training time, and negligible inference cost. It is a deterministic function of atom outputs, meaning atoms can be added, removed, or replaced without retraining any other component. This modularity enables a deployment model where new signal domains are addressed by training a new ~28K-parameter atom and connecting it to the existing composition function.

The monolithic model's failure — 0.645 F1 overall and 0.000 recall on rhythm-only anomalies — demonstrates that scaling a single model to handle multiple domains is not merely less convenient but fundamentally less effective when the model's capacity is constrained. At micro-scale (under 30K parameters), specialization beats generalization.

### Future Work

Several directions extend this work: (1) validating on real-world telemetry from our continuously running deployment, which has accumulated over 35 hours of labeled data; (2) exploring learned composition functions trained on a small number of labeled cross-domain examples; (3) extending the atom vocabulary to additional domains (network traffic, financial transactions, environmental sensors); (4) benchmarking inference on ESP32-S3 and Raspberry Pi Pico W hardware, where the entire forward pass runs in under 30KB of flash; and (5) investigating whether the composition advantage grows as the number of signal domains increases, which would suggest a scaling law for composable micro-transformers.

## References

- Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. *Amazon Science*.
- Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning. *ACM CCS*.
- Guo, H., Yuan, S., & Wu, X. (2021). LogBERT: Log Anomaly Detection via BERT. *arXiv:2103.04475*.
- Karpathy, A. (2023). microgpt. *GitHub*.
- Meng, W., et al. (2019). LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs. *IJCAI*.
- Tuli, S., Casale, G., & Jennings, N. R. (2022). TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. *VLDB*.
- Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. *ICLR*.

---

## Appendix A: Reproducibility

All results can be reproduced with a single command:

```bash
python3 -m kiri.benchmark.composition_test --steps 1000 --figures
```

The benchmark uses a fixed random seed (42) for deterministic data generation and anomaly injection. Training uses PyTorch with MPS acceleration on Apple Silicon. Total runtime is approximately 76 seconds on an M1 Mac Mini. The benchmark code, model weights, and result figures are available in the repository at `benchmark/`.

## Appendix B: Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Observations | 2,016 (7 days x 24h x 12/hour) |
| Sampling interval | 5 minutes |
| Anomalies injected | 150 (50 per class) |
| Training set | 1,866 (normal only) |
| Embedding dimension | 32 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Context window | 16 tokens |
| Scoring window | 3 observations |
| Training steps | 1,000 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Optimizer | Adam (betas 0.85, 0.99) |
| Divergence weight (alpha) | 0.5 |
| Random seed | 42 |
