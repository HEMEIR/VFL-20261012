# SOTA Defenses Against Label Inference Attacks in Two-Party VFL (PMC / AMC)

Research brief for the 2026 two-party VFL competition, constrained to the three
`configurePartyADefenseHooks` extension points.

**Epistemic convention used throughout.**
- **[VERIFIED-CODE]** — established by reading the locked files in this repository.
- **[VERIFIED-PAPER]** — stated in the cited paper's own text (HTML full text fetched).
- **[SNIPPET]** — seen only in a search-result snippet, not in fetched full text.
- **[INFERENCE]** — my own reasoning. Not validated by experiment.
- **[UNVALIDATED]** — I could not run any experiment: this machine has **no PyTorch**
  (only the bundled runtime, which ships numpy/pandas but not torch or sklearn) and the
  shell has no working outbound TLS. No claim below is backed by a run of `main.py`.

---

## 0. Headline conclusions

1. **The three hooks defend LFBA/VILLAIN but are fundamentally mismatched to PMC/AMC.**
   PMC finetunes a **copy of B's own bottom model on B's own features**, so no
   inference-time hook is ever on its path. Only *training-time* effects on B's learned
   parameters matter. **[VERIFIED-CODE]**

2. **The single genuinely available lever is the `cutGradientHook`.** It is the only thing
   A controls that changes B's actual learned bottom model. **[VERIFIED-CODE]**

3. **The current SOTA anti-MC defense, VMask, is INFEASIBLE under these rules**, because it
   masks layers of the *passive party's* bottom model. **[VERIFIED-PAPER]**

4. **The competition's auxiliary label set is tiny and perfectly balanced: 10 samples per
   class (20 rows for binary, 30 for the 6-class dataset).** **[VERIFIED-CODE]** This is the
   attack's soft spot: 10 anchors per class plus a λU = 50 unlabeled self-training objective
   is an extremely brittle alignment procedure. **[INFERENCE]**

5. **The best-fitting literature defense is the information-theoretic family** (MID / VIB,
   distance correlation, CoAE entropy regularization), because it lowers
   `I(cutB ; Y)` — which provably caps *any* head the attacker trains, including one that
   generalizes to the hidden test set — rather than playing games with the 20-sample anchor
   set. **[VERIFIED-PAPER + INFERENCE]**

---

## 1. The exact competition system profile

### 1.1 The attack, as actually implemented here

- **[VERIFIED-CODE]** `attacks/label_inference/model_completion.py:82-104` — `PassiveCompletionModel`
  holds `self.bottomModel = copy.deepcopy(bottomModel)` and an `inferenceHead`
  (`CompletionHead`) of `headLayers` Linear layers (`configs/dataset_03.yaml` locks
  `headLayers: 1`, `useBatchNorm: true`, activation `relu`), each preceded by `BatchNorm1d`.
- **[VERIFIED-CODE]** `model_completion.py:421-546` — `runPassiveModelCompletion` is invoked
  **after joint VFL training with the final `bottomB`**. It reads labels *only* from
  `auxiliaryLabelsPath` and never touches A's features or the top model.
- **[VERIFIED-CODE]** `model_completion.py:250-361` — `trainPmcEpoch` is a custom MixMatch:
  a labeled batch of size `completionBatchSize` from the auxiliary set, plus unlabeled
  batches drawn from B's own features; pseudo-labels are temperature-sharpened and MixUp'd;
  `loss = labeledLoss + unlabeledWeight * unlabeledLoss` with `unlabeledWeight` ramping to
  `lambdaU` (= 50.0 in the PMC scenario config).
- **[VERIFIED-CODE]** `model_completion.py:364-385` — `predictCompletionModel` calls
  `model(batch["partyB"])` directly. **The `inferenceRepresentationHook` is never invoked on
  the attacker's path.**
- **[VERIFIED-CODE]** `model_completion.py:169-175` — auxiliary IDs are *training* IDs
  (`TRAIN_*`), disjoint from the hidden test IDs (`TEST_*`). So the attack must **generalize**
  from the 20–30 anchors to held-out samples. **[INFERENCE]** This is the property any
  defense should target.

### 1.2 The defender's actual levers

| Hook | Signature (from `core/trainer.py:100-207`) | What it can change |
|---|---|---|
| `trainingRepresentationHook` | `(cutA, cutB, labels, sampleIds, epoch, batchIndex, context, generator) -> (cutA, cutB)` | What the top model sees; shapes B **through backprop** |
| `cutGradientHook` | `(gradientB, passiveCut, labels, sampleIds, epoch, batchIndex, context, generator) -> gradientB` | The gradient B applies to its own bottom model |
| `inferenceRepresentationHook` | `(embeddingA, embeddingB, sampleIds, context, generator) -> (embeddingA, embeddingB)` | A's two-party inference only |

Key plumbing facts **[VERIFIED-CODE]**:
- `trainer.py:423-424`: `cutA`, `cutB` are **detached** with `requires_grad_(True)`. Gradients
  reach A's and B's bottom models **only** through these two tensors. `trainer.py:437-440`
  raises if a hook detaches the path.
- `trainer.py:444-456`: `gradientB` is cloned, passed through `cutGradientHook`, then used as
  `embeddingB.backward(gradientB)`.
- `trainer.py:399` + `:187`: **`sampleIds` are available in both the training and the
  inference hooks.** This is essential (see §4).
- `trainer.py:457` / `active_model_completion.py:45-123`: AMC's
  `AdaptiveMaliciousGradientScaler` acts on `parameter.grad` of **B's bottom model** *after*
  `embeddingB.backward(gradientB)`, and clamps its per-coordinate factor to
  `[minimumGradientScale, maximumGradientScale]` = **[1.0, 5.0]**. It normalizes by the
  *previous unscaled* direction, so it is a **relative** amplification.

### 1.3 Why the existing defense is inert against PMC — the repo says so itself

`defense/robust_vfl.py:24-26` records, in the authors' own measured words:

> 梯度加噪对标签恢复率几乎无改善（本项目的 PMC 只用 B 方本地特征与辅助标签训练补全模型，
> 并不消费 A 方返回的 cut 梯度）

("[gradient noise] gives almost no improvement in the label-recovery rate (this project's PMC
trains its completion model using only B's local features and auxiliary labels, and does not
consume the cut gradient A returns)".) **[VERIFIED-CODE]**

Note the subtlety this comment misses, which §4 exploits: PMC does not *consume* the cut
gradient, but the cut gradient determines **which parameters B ends up with**, and PMC then
copies those parameters. So the cut gradient is still the lever — just an indirect one.

### 1.4 Measured geometry of the five datasets (this matters a lot)

Full-batch logistic-regression probes on the raw CSVs (own code, numpy only, standardized
features). **[VERIFIED-MEASUREMENT — mine, by script, on the real data]**

| dataset | n train | classes | A feats | B feats | A-only probe | B-only probe | A+B probe | A-only shortfall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dataset_01 | 8,400 | 2 | 1536 | 1536 | 0.550 | 0.552 | 0.570 | 0.020 |
| dataset_02 | 43,489 | 2 | 634 | 1000 | 0.799 | **0.959** | 0.977 | **0.178** |
| dataset_03 | 7,738 | 2 | 15 | 15 | **0.920** | 0.698 | 0.927 | **0.007** |
| dataset_04 | 34,190 | 2 | 7 | 7 | 0.784 | 0.803 | 0.826 | 0.042 |
| dataset_05 | 7,209 | 6 | 280 | 281 | **0.989** | 0.972 | 0.990 | **0.001** |

Interpretation **[INFERENCE]**:
- **dataset_03 and dataset_05 are the favorable cases**: A's own features already carry
  essentially all the label signal. Degrading B's contribution costs ≈0.1–0.7 points of
  headroom. dataset_03 is the ideal testbed — a *targeted* design target.
- **dataset_02 is the hostile case**: B holds the bulk of the signal (0.959 vs A's 0.799).
  Any defense that degrades B's embedding will cost real accuracy here. Do not over-tune on
  dataset_03 and expect dataset_02 to survive.
- dataset_01's low absolute probes (both ≈0.55) with 1536 features per party suggests
  image-like, non-linearly-separable data; linear probes understate achievable NN accuracy
  there, so read that row as "both parties similar", not "task is hard".

Auxiliary set composition **[VERIFIED-MEASUREMENT]**: `dataset_01/02/03/04_pmc.csv` = 20 rows
(10 per class); `dataset_05_pmc.csv` = 30 rows (10×6). Perfectly balanced in every case.

---

## 2. PMC / AMC: exact attack mechanisms

**Fu, Zhang, Ji, Chen, Wu, Guo, Zhou, Liu, Wang. "Label Inference Attacks Against Vertical
Federated Learning." USENIX Security 2022, pp. 1397–1414.**
[USENIX page](https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong) ·
[paper PDF](https://www.usenix.org/system/files/sec22-fu-chong.pdf) ·
[slides](https://www.usenix.org/system/files/sec22_slides-fu-chong.pdf)

I could not fetch the PDF (the fetch tool rejects `application/pdf` and the shell has no
working TLS), so I reconstruct the mechanism from three independent secondary descriptions
that agree, and flag the residual uncertainty.

**PMC (Passive Model Completion).** The passive party B uses its **locally trained bottom
model** `f(θ_k)` as a feature extractor, attaches an **inference head** `f(θ_H)` with the same
output dimension as A's top model, and **fine-tunes the combined model `f_A = f(θ_k) ∘ f(θ_H)`
with semi-supervised learning** on an attack dataset consisting of `M` samples for which B
somehow holds labels plus `N−M` unlabeled samples. B then infers the label of any sample from
its own features. **[VERIFIED-PAPER via [VMask §II-B](https://ar5iv.labs.arxiv.org/html/2507.14629)]**

**AMC (Active Model Completion).** Same completion step, but *before* it the attacker
**actively adapts its local bottom-model updates during VFL training** so that the embedding
becomes more label-informative, then runs the PMC completion step. **[VERIFIED-PAPER via
[Zou et al. §3.2](https://ar5iv.labs.arxiv.org/html/2301.01142): "Depending on whether the
attacker updates its local model actively to infer more information, MC attack can be
separated into active MC attack (AMC) and passive MC attack (PMC)"]**

**Attack capability table** (from [Liu et al., arXiv:2112.05409, Table 1](https://ar5iv.labs.arxiv.org/html/2112.05409)):
PMC and AMC **do not require sample-level gradients** but **do require auxiliary labeled
samples**; they support ≥2 classes. This is the decisive structural fact: *PMC's information
bottleneck is the auxiliary set, not the gradient channel.*

**Defenses Fu et al. themselves suggested**: perturbing the backpropagated gradients;
discrete gradients; gradient sparsification. **[VERIFIED-PAPER via VMask §I: "Fu et al. [9]
propose perturbing the backpropagated gradients to reduce the information in it"] and via
[MID §2](https://ar5iv.labs.arxiv.org/html/2301.01142), which lists "Gradient Sparsification
(GS)", "Discrete Gradients (DG)", and adding noise as the defenses Fu et al. evaluated.**
⚠️ **[UNVERIFIED]** — I could not read Fu et al.'s own §"possible defenses" or their reported
numbers for these; treat the specific values as unconfirmed.

---

## 3. Candidate defenses: inventory

Legend for **Feasibility**: which hook can carry it, or *infeasible*.

### 3.1 VMask — layer masking via secret sharing  ★ current SOTA against MC
- **Tan, Zhang, Hu, Yang, Ran, Li. "VMask: Tunable Label Privacy Protection for Vertical
  Federated Learning via Layer Masking." NDSS 2025; extended in Front. Comput. Sci. 21(2)
  2102325, 2027.** [arXiv:2507.14629](https://arxiv.org/abs/2507.14629) ·
  [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2507.14629) ·
  [FCS version](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-51046-z)
- **Mechanism [VERIFIED-PAPER]**: randomize ("mask") a selected subset of **the attacker's
  bottom-model linear layer parameters** using additive secret sharing, and train the masked
  layers *under SS* so linear computation stays exact. A **shadow model** (same architecture as
  B's bottom model, trained synchronously by A on a small auxiliary dataset) lets A locally
  simulate the MC attack each epoch and select layers by **accumulated gradient norm**
  `G_k^j = Σ_t ‖∇θ^j_{k,t}‖` until estimated leakage drops below a privacy budget.
- **Where it sits**: **inside the passive party's bottom model** — i.e. it requires the
  *protocol* to force masking on B.
- **Reported effectiveness [VERIFIED-PAPER]**: "reducing the label inference accuracy to a
  random guessing level while preserving model performance (e.g., in Transformer-based model,
  the averaged drop of VFL model accuracy is only 0.09%)"; "average model accuracy drop of at
  most 0.34% in all cases"; 13 datasets, 5 architectures (MLP3, LeNet5, VGG13, ResNet18,
  DistilBERT); compared against 12 other defenses; up to 60,846× faster than HE-based methods.
  Single-layer masking (LeNet5/MNIST): masking layer 1 → attack acc 0.5481 ± 0.0384 vs
  masking layer 5 → 0.8373 ± 0.0148. VGG13/CIFAR10: first layer → 0.3902 ± 0.0748, all layers
  → 0.2026 ± 0.0129. Non-adjacent masking beats adjacent (ResNet18/CIFAR10: (θ1,θ3) → 0.2771 ±
  0.0419 vs (θ1,θ2) → 0.5043 ± 0.0895). **[SNIPPET]** a results table sampled "VGG13 | CF10 |
  CN10 | 90.74 | 70.79 | 90.64 | 73.62 | 14.46 | 15.55" — consistent with MC accuracy collapsing
  from ~90% to ~15%, but I could not verify the column semantics in full text.
- **Feasibility here: INFEASIBLE.** The rules forbid modifying B's bottom model
  (`README.md:171-177`, `defense/robust_vfl.py` has no route to B's parameters). **This is the
  most important negative result in this brief** — the best-known method is unavailable, so the
  competition is genuinely asking for a new class of defense.

### 3.2 MID — Mutual Information Regularization / Variational Information Bottleneck
- **Zou, Liu, Zhang. "Mutual Information Regularization for Vertical Federated Learning."**
  [arXiv:2301.01142](https://ar5iv.labs.arxiv.org/html/2301.01142) (also IEEE:
  [xplore 10825545](https://xplorestaging.ieee.org/document/10825545))
- **Mechanism [VERIFIED-PAPER]**: A inserts `M_VIB` at the cut. It treats the network as the
  Markov chain `Y − X^p − H^p − T − Z − Ŷ` with stochastic encoding layer `T`, and by the Data
  Processing Inequality `I(H^p; Ŷ) ≤ I(H^p; T)`. Objective:
  `min_T { −I(Y;T) + λ·I(H^p;T) }`, combined with the task loss:
  `L = ℓ(S(H^a, Z), Y) + λ·I(H^p, T)`, with `Z = M_VIB(H^p)`. Because exact MI is intractable,
  `M_VIB` is implemented as a **Variational Information Bottleneck**: an encoder producing
  `(μ, σ)` with a reparameterized sample for `T`, plus a decoder. Multi-party:
  `Σ_k λ^k I(H^k, T^k)`.
- **Where it sits**: at the **active party**, between B's embedding and the top model. It
  **transforms the representation the top model consumes** — exactly the shape of
  `trainingRepresentationHook`.
- **Reported effectiveness [VERIFIED-PAPER]**: claimed "better utility-privacy trade-off than
  other general defense methods for various feature reconstruction attacks, label inference
  attacks and backdoor attacks", with theoretical MI guarantees. **⚠️ [UNVERIFIED]** — I could
  not extract the numeric tables (they are in figures I cannot read). VMask's §I characterizes
  this family as "either achieve good model performance or strong data privacy, but rarely both
  simultaneously". **[VERIFIED-PAPER]** for the characterization, not for MID's own numbers.
- **Feasibility here: HIGH for the *idea*, MEDIUM for the letter.** The paper's `M_VIB` is a
  *new module inserted between H^p and the top model*, which the rules permit ("A may add
  auxiliary defense modules... must not replace or extend the main-task bottom model or top
  model, and must not change their agreed input/output interfaces"). But MID's own protocol
  inserts it into the architecture and requires its parameters to be trained by the VFL
  objective. Under the hooks, we get a **differentiable** route to train such a module — see §4.3.
  MID defends **both PMC and AMC**, because it caps `I(cutB; Y)` and therefore caps what *any*
  head, trained by *any* procedure, can recover — including one that generalizes to test data.

### 3.3 Distance-correlation defense (forward-embedding leakage)
- **Sun, Yang, Yao, Wang. "Label Leakage and Protection from Forward Embedding in Vertical
  Federated Learning."** [arXiv:2203.01451](https://ar5iv.labs.arxiv.org/html/2203.01451)
- **Mechanism [VERIFIED-PAPER]**: A adds an extra optimization target that **minimizes the
  distance correlation `dCor(f(X), Y)`** between the cut-layer embedding and the labels. The
  authors compute pairwise distance matrices `s_(j,k) = ‖f(X)_j − f(X)_k‖` and
  `t_(j,k) = ‖Y_j − Y_k‖`, double-center them, and minimize the resulting normalized
  dependence measure. It is optimized jointly with cross-entropy, so it fits the ordinary
  optimization framework.
- **Where it sits**: **at the label party**, as an added term in the training objective. This is
  the closest published match to "A modifies A's own loss function".
- **Reported effectiveness [VERIFIED-PAPER]**: their **spectral attack** on the forward
  embedding reaches leak AUC ≈ 0.78 (vanilla, last two layers) and **still ≈ 0.75 under
  Marvell (s=8.0) and under Label DP (ε=0.5)** — i.e. *gradient-side defenses do nothing
  against embedding-side attacks*. They also establish that **leak AUC is bounded by the model's
  test AUC**. **[UNVERIFIED]** — I could not extract the numeric leak-AUC under their own
  distance-correlation defense; only the qualitative claim of effectiveness.
- **Feasibility here: HIGH.** `dCor(cutB, labels)` is computable inside
  `trainingRepresentationHook` (it receives `cutB` and `labels`), and §4.3 shows how to make
  the transformation trainable so the objective is actually optimized.

### 3.4 CoAE / CAE — autoencoder + entropy regularization ("confusional autoencoder")
- **Liu, Zou, Kang, Liu, He, Yi, Yang. "Batch Label Inference and Replacement Attacks in
  Black-Boxed Vertical Federated Learning."** [arXiv:2112.05409](https://ar5iv.labs.arxiv.org/html/2112.05409)
  (Zou et al., "Defending Label Inference and Backdoor Attacks in Vertical Federated Learning",
  AAAI 2022, is the other CAE reference — **[UNVERIFIED]**, I could not fetch it.)
- **Mechanism [VERIFIED-PAPER]**: an autoencoder plus **entropy regularization** placed at the
  **active party**, to "hide the true labels without any noticeable changes to the original
  protocol"; it transforms the messages rather than the model. They report it blocks label
  inference and replacement attacks "without hurting as much main task accuracy as compared to
  existing methods like differential privacy (DP) and gradient sparsification (GS)".
  Their batch label inference attack (gradient inversion on batch-averaged gradients) reaches
  **>90% accuracy**, and Theorem 1 states labels can be **100% recovered** when
  `B ≤ rank(A) ≤ M`.
- **Feasibility here: HIGH conceptually.** "Hide labels at the active party by transforming the
  cut representation, with an entropy term that destroys the representation→label mapping" is
  implementable in `trainingRepresentationHook`. Defends **both PMC and AMC** for the same
  information-theoretic reason as MID. **[UNVERIFIED]** numbers for CoAE against *model
  completion* specifically — it was designed against batch-level gradient inversion.

### 3.5 Marvell and the norm/direction-scoring defenses
- **Li, Sun, Yang, Gao, Zhang, Xie, Smith, Wang. "Label Leakage and Protection in Two-party
  Split Learning." ICLR 2022**, [arXiv:2102.08504](https://ar5iv.labs.arxiv.org/html/2102.08504)
  (often cited as "Munich/ICML 2022" — the correct venue is **ICLR 2022**; flag as a
  venue correction.)
- **Mechanism [VERIFIED-PAPER]**: two attacks — a **norm-based** scoring function
  `r_n(g) = ‖g‖₂` (exploiting `‖g‖₂ = |p̃₁ − y|·‖∇_z h(z)‖₂`, with positive-class confidence gaps
  systematically larger), and a **direction-based** scoring function `r_d(g) = cos(g, g₊)`
  (exploiting `cos(g_a, g_b) = sgn(p̃₁,a − y_a)·sgn(p̃₁,b − y_b)·cos(∇_z h, ∇_z h)` together with
  the observation that the two gradient directions always form an acute angle). They define
  **leak AUC**. **Marvell** then "strategically finds the structure of the noise perturbation by
  minimizing the amount of label leakage ... of a worst-case adversary", combining a
  norm-alignment heuristic (match expected squared 2-norm of every perturbed gradient in the
  minibatch to the largest, `‖g_max‖₂²`) with a rank-1, example-dependent directional noise
  `Cov[η_j] = σ_j² g_j g_jᵀ`. They note DP is not applicable because sample-level gradients are
  example-specific and cannot be averaged or shuffled.
- **Reports leak AUC > 0.9 for the norm attack and = 1.0 for the oracle cosine attack without
  protection [VERIFIED-PAPER].**
- **What it defends: NOT PMC/AMC.** Marvell protects the **cut-layer gradient** channel. PMC/AMC
  never read that channel. Decisively, [Sun et al.](https://ar5iv.labs.arxiv.org/html/2203.01451)
  show the *embedding-side* spectral attack survives Marvell at s=8.0 with leak AUC ≈ 0.75.
  Since PMC attacks the embedding side, Marvell is the wrong tool. **[VERIFIED-PAPER +
  INFERENCE]**

  **Literature corroboration of this classification [VERIFIED-PAPER]:** in
  [arXiv:2112.05409](https://ar5iv.labs.arxiv.org/html/2112.05409), the authors' Table 1 compares
  label-inference attacks and marks PMC/AMC as *not* requiring sample-level gradients, while the
  norm-based and direction-based scoring attacks (the ones Marvell targets) *do* require them.
  Consistently, in the [VMask](https://ar5iv.labs.arxiv.org/html/2507.14629) related-work
  section, Marvell is **never listed as a defense against MC** — the anti-MC defenses named
  there are gradient perturbation (Fu et al.), autoencoder/DP (Zou et al., Ghazi et al.),
  correlation-reducing regularization (Zou/Sun/Zheng), and cryptography. So across the surveyed
  literature, **no paper claims Marvell defends PMC/AMC.**

### 3.6 Gradient compression / sparsification / discretization; noisy & DP gradients
- References: [MID §2](https://ar5iv.labs.arxiv.org/html/2301.01142) (Gradient Sparsification,
  Discrete Gradients, adding noise, gradient compression); [FedAdOb §II-B](https://ar5iv.labs.arxiv.org/html/2406.01085)
  ("Sparsification compresses disclosed [information] to mitigate privacy leakage", citing
  Fu et al. and Lin et al.); Zhao et al. 2020 and Zhu et al. 2019 (gradient inversion).
- **[VERIFIED-PAPER]** VMask §I groups these as ML-based anti-MC defenses that fail on
  privacy-utility trade-off (challenge C1). FedAdOb states "all the aforementioned defense
  mechanisms experience a certain level of performance degradation" and attributes this to
  *fixed* obfuscation.
- **Feasibility here: partial, and weak against PMC.** A DP-style perturbation of
  `gradientB` is exactly the `cutGradientHook`. But it only perturbs **directions/magnitudes of
  B's parameter updates**; with a λU = 50 self-training objective that drowns the 10-per-class
  anchor signal, the completion model is already robust to update noise. This matches the
  repository's own measurement (`defense/robust_vfl.py:24-26`). **[VERIFIED-CODE]**

### 3.7 Representation obfuscation / disentanglement / adversarial training
- **[VERIFIED-PAPER]** VMask §I lists the "extra regularization term to reduce the correlation
  between intermediate information and private labels" line of work as Zou et al. [13],
  Sun et al. [8] (the distance-correlation defense, §3.3), and Zheng et al. [14].
- **[VERIFIED-PAPER]** FedAdOb: **Gu, Luo, Kang, Yao, Zhu, Li, Fan, Yang. "FedAdOb:
  Privacy-Preserving Federated Deep Learning with Adaptive Obfuscation."**
  [arXiv:2406.01085](https://ar5iv.labs.arxiv.org/html/2406.01085). Mechanism: **passport-based
  adaptive obfuscation** `g_W(x_in, s) = γ(W x_in) + β` with
  `γ = Avg(D(E(W s^γ)))`, `β = Avg(D(E(W s^β)))`, where `s^γ, s^β` are secret random passports
  (`s(j) ~ N(μ_j, σ²)`, `μ_j ~ U(−N,0)`) and `E, D` are an autoencoder. The scale/bias are
  *learned in tandem* with the model, which is what makes it preserve accuracy where fixed
  obfuscation does not. Crucially, **FedAdOb handles PMC** — its Table I threat model lists
  PMC [7] for the VFL semi-honest passive party — by embedding passports in the **top model**
  (label protection) and the **bottom models** (feature protection). Theorems 1 and 2 prove
  hardness of feature and label restoration.
- **Feasibility here: PARTIAL.** The *label-protection half* (passport in the top model) is
  conceptually available via the representation hook. The *feature-protection half* (passport
  inside B's bottom model) is forbidden, same reason as VMask.

### 3.8 Dimensionality reduction / dimensional privacy / cut-layer position
- **Liu, Xu, Guo, Cao, Wang. "Revisiting Label Inference Attacks in Vertical Federated
  Learning: Why They Are Vulnerable and How to Defend."** [arXiv:2603.18680](https://arxiv.org/abs/2603.18680)
  (submitted 19 Mar 2026) · [ar5iv](https://ar5iv.labs.arxiv.org/html/2603.18680)
- **Findings [VERIFIED-PAPER]**: they prove `I(X^i;Y) ≤ I(T_1^i;Y) ≤ ... ≤ I(T_n^i;Y) ≤ I(S_1;Y)`,
  i.e. mutual information with the label **increases with layer depth**, so "bottom models
  primarily extract feature information while the top model handles label mapping". The ratio of
  top-model to bottom-model maximum MI ranges **13.17 to 306.92** across five datasets (MNIST,
  CIFAR-10, CINIC-10, SVHN, GTSRB) and five architectures. They introduce **task reassignment**:
  deliberately disrupting the latent feature↔label alignment makes LIA accuracy "decline sharply
  or even fail entirely", concluding that existing LIAs' success "is largely an illusion, heavily
  dependent on a strong natural alignment between sample features and labels".
- **Defense [VERIFIED-PAPER]**: "**zero-overhead** defense technique based on layer adjustment" —
  **shifting the cut layer forward** to increase the top model's share of the network. Advancing
  by one layer gives substantial benefit; by three layers reduces LIA accuracy to random
  guessing; **and it improves predictive performance**.
- **Feasibility here: INFEASIBLE** (moving the cut layer changes the agreed model split).
- **BUT this is the most useful paper in the brief for our purposes [INFERENCE]**: the
  "task reassignment" result is direct evidence that **breaking the auxiliary feature↔label
  alignment** is a viable defense principle, and it explains why the anchor-manipulation idea
  in §4.1 should work. It also gives the best available argument for the *information-theoretic*
  defenses: if bottom models carry little MI with the label, LIAs have no ceiling to exploit.

### 3.9 Label compression
- **"Defend against Label Inference Attacks in Vertical Federated Learning via Label
  Compression."** [IEEE Xplore 11435639](https://ieeexplore.ieee.org/abstract/document/11435639)
  · code: [github.com/YuanShunJie1/LCD](https://github.com/YuanShunJie1/LCD)
- **[UNVERIFIED]** — the IEEE page returned only account boilerplate and the GitHub page
  returned only site chrome; I could not extract the mechanism or any numbers. The title
  indicates compressing the label information available to the attacker, which is the right
  *shape* of idea, but I will not speculate about its mechanics.

### 3.10 Label DP / randomized response
- **[VERIFIED-PAPER]** via [Sun et al.](https://ar5iv.labs.arxiv.org/html/2203.01451): "some
  recent work [8] leveraged randomized responses to flip labels and use the generated noisy
  version labels to compute the loss functions. They proved that their proposed algorithms can
  achieve label differential privacy (DP)."
- **Feasibility here: MODERATE.** Flipping a fraction of training labels inside A's own loss
  computation is "modifying A's own loss / A's own labels", which is explicitly allowed. It
  directly attacks the feature↔label alignment that [2603.18680](https://ar5iv.labs.arxiv.org/html/2603.18680)
  identifies as the root cause. **But** label noise degrades *B's* learned features too (B's
  gradient comes from the noisy loss), so the accuracy cost is nontrivial, and Sun et al. show
  **label DP alone leaves forward-embedding leakage at leak AUC ≈ 0.75**. **[VERIFIED-PAPER]**

---

## 4. Mapping every candidate onto the three hooks

The decisive asymmetry, restated precisely **[VERIFIED-CODE]**:

> During the completion phase, `PassiveCompletionModel.forward` is
> `inferenceHead(bottomModel(partyB))` with `bottomModel` a **deepcopy of B's final parameters**.
> It never calls into the trainer. Therefore **nothing A does at inference time can affect
> PMC/AMC.** The only causal channel is: *(hooks applied during joint training) → (B's learned
> θ_B) → (the parameters PMC copies)*.

Two corollaries:

**(a) The `trainingRepresentationHook` is not just for the top model.** Because `cutB` carries
gradient back into `embeddingB` (`trainer.py:456`), any transformation of `cutB` changes the
gradient B receives and therefore changes `θ_B`. A transformation that is *ondemand* on
`sampleIds` can shape `θ_B` differently for the auxiliary samples than for the rest.
**[VERIFIED-CODE]**

**(b) The `inferenceRepresentationHook` is a pure accuracy-protection tool for PMC/AMC.** Its
only value is to *undo* a training-time manipulation at test time so the main task is not
harmed. This is why the auxiliary-anchor idea in §4.1 is architecturally plausible: **manipulate
at train time, restore at inference time.** But note the trap: during *training* the top model
sees the manipulated `cutB` while at *inference* it sees the restored one, so any manipulation
that the top model needs to compensate for creates a train/test mismatch. Keep the manipulation
small in the main-task-relevant directions and large only in the *anchor-identification*
directions. **[INFERENCE]**

**(c) The `cutGradientHook` is the only direct lever on `θ_B` and it is under-used.**
**[VERIFIED-CODE]** The shipped defense sets `gradientNormalizeStrength: 0.0`,
`gradientNoise: 0.0`, `gradientClassCenteringStrength: 0.0`. Two facts make this interesting:
- A **downward** scaling of `gradientB` (e.g. ×0.05) reduces B's effective learning rate. A
  bottom model that barely moved from a weak/random initialization is a weak feature extractor,
  which is precisely the regime VMask emulates by randomizing B's layers. **[INFERENCE]**
- AMC's scaler clamps its factor to **[1.0, 5.0]** (`maximumGradientScale: 5.0`,
  `minimumGradientScale: 1.0`). It can therefore **amplify by at most 5×**. A downward scale
  with `s ≪ 1/5` is **not recoverable by AMC**. This is a concrete, checkable asymmetry between
  the passive and active attacks. **[VERIFIED-CODE + INFERENCE]**

**(d) `gradientClassCenteringStrength` targets exactly the wrong quantity but in the right
place.** The shipped `_centerClassConditional` removes each class's mean gradient direction.
Class-mean removal does reduce the label-discriminative component B can learn — but it also
removes the signal the *top model* needs from B, and the repo measured that it "把 dataset_03
clean 打到 0.59" (drops dataset_03 clean accuracy to 0.59). **[VERIFIED-CODE]** Too blunt.

**(e) What is NOT available at all**: modifying B's architecture or per-layer parameters
(VMask, FedAdOb's bottom passport), moving the cut layer (2603.18680), and anything requiring
HE/MPC/TEE (out of protocol scope).

---

## 5. Ranked shortlist — 5 most promising, directly applicable defenses

Ranked by (feasibility within the hooks) × (expected PMC+AMC reduction) ÷ (main-task cost),
with the dataset_02 hostile case always in mind.

### R1. Cut-gradient attenuation with AMC-proof downward scaling — *the highest-confidence, lowest-risk lever*
**Why it should work.** It is the only mechanism that directly changes `θ_B` — the sole thing
PMC copies — and it exploits a hard cap that AMC cannot exceed (factor ∈ [1,5]).
**[VERIFIED-CODE + INFERENCE]**

**Where.** `cutGradientHook` only.

**Pseudocode.**
```
# config: cutGradientScale, rampEpochs, perClassScale (optional)
def cutGradientHook(gradientB, passiveCut, labels, sampleIds, epoch, batchIndex,
                    context, generator, self):
    if self.cutGradientScale >= 1.0:
        return gradientB                       # no-op, keep baseline behavior
    # ramp in so early joint training still builds a usable B; then attenuate
    t = min(1.0, (epoch + 1) / self.rampEpochs)
    s = 1.0 - t * (1.0 - self.cutGradientScale)   # s: 1.0 -> cutGradientScale
    g = gradientB * s
    # optional: attenuate harder for the auxiliary samples' IDs (A knows them from the
    # public CSV), leaving the bulk of B's training signal intact
    if self.auxIds is not None:
        mask = torch.tensor([sid in self.auxIds for sid in sampleIds],
                            device=g.device).unsqueeze(1)
        g = torch.where(mask, g * self.auxGradientScale, g)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)   # must stay finite
```
**Sweep.** `cutGradientScale ∈ {0.5, 0.2, 0.1, 0.05}`; `rampEpochs ∈ {10, 30}`. Below ≈0.1
expect B to stay near initialization.

**Attacks defended.** **Both.** PMC gets a weaker feature extractor to complete; AMC cannot
amplify its way out because its factor is capped at 5.

**Honest tradeoff [INFERENCE].** This is a *blunt, information-reducing* instrument. It lowers
B's contribution to the joint model. Expect:
- dataset_03, dataset_05: **near-free** (A-only probes 0.920 and 0.989 vs joint 0.927/0.990 —
  ≤0.1–0.7 points of headroom).
- dataset_04: **mild** (0.784 → 0.826, 4.2 points of headroom).
- dataset_02: **expensive** (0.799 → 0.977; B carries 17.8 points). At scale 0.1 this could cost
  5–15 points of clean accuracy. Consider a per-dataset scale, but check whether the rules allow
  dataset-dependent defense parameters (the defense config is global by default —
  `README.md:188-199`). If not, this is the main risk of R1.
- AMC's own interaction: because AMC amplifies *after* the hook, a scale `s` yields an effective
  B update of `≤ 5s`. At `s = 0.05` even maximal AMC amplification gives ≈0.25× the clean step.

### R2. Trainable VIB / distance-correlation bottleneck on `cutB` (MID + dCor), trained through the top-model path
**Why it should work.** It targets the *information* `I(cutB; Y)`, which caps label recovery for
**any** head the attacker trains — so it transfers to the hidden test set, unlike any trick
confined to the 20 anchors. Two independent papers support the principle, and
[2603.18680](https://ar5iv.labs.arxiv.org/html/2603.18680) supplies the mechanism-level argument
(bottom-model MI vs label rises with depth; disrupting feature↔label alignment kills LIAs).
**[VERIFIED-PAPER + INFERENCE]**

**Where.** `trainingRepresentationHook` + `inferenceRepresentationHook` (shares the same module).

**How the module gets trained — the key implementation insight [INFERENCE].** The hooks have no
loss-injection channel, but they do not need one: make the transformation **differentiable and
parameterized**, and the ordinary task loss `ℓ(topModel(cutA, h_φ(cutB)), y)` backpropagates
into `φ` automatically. The *regularizer* is then added in one of two legitimate ways:
(i) as an **entropy/whitening penalty on the module's own bottleneck statistics** computed
inside the hook, which biases `φ` toward a low-information encoding even without a separate
backward pass; or (ii) by exposing the module to A's optimizer (`self.optimizerA` /
a new optimizer step in the hook) using a locally computed `dCor`/VIB-KL term. Option (ii)
requires a local backward pass inside the hook, which is legal — it is A's own training logic.

**Pseudocode.**
```
class RepBottleneck(nn.Module):           # auxiliary module - allowed by the rules
    def __init__(self, d, k, beta=1e-3):
        self.mu  = nn.Linear(d, k); self.logvar = nn.Linear(d, k)
        self.dec = nn.Linear(k, d)        # decoder -> keeps interface shape == d
        self.beta = beta
    def forward(self, h, generator):
        mu, lv = self.mu(h), self.logvar(h).clamp(-8, 8)
        if self.training:
            eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype,
                              device=mu.device)
            t = mu + eps * (0.5 * lv).exp()
        else:
            t = mu                       # deterministic at inference
        self.lastKL = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean())
        return self.dec(t)               # shape == cutB shape, dtype/device preserved

def trainingRepresentationHook(cutA, cutB, labels, sampleIds, epoch, batchIndex,
                               context, generator, self):
    z = self.bottleneck(cutB)
    # optional distance-correlation term, differentiable, same batch
    if self.dcorStrength > 0 and labels is not None and z.shape[0] >= 4:
        self.lastDcor = distanceCorrelation(z, labels.float())
    # annealed regularizer objective for A's own optimizer step:
    #   L_def = beta * KL(q(t|h) || N(0,I)) + dcorStrength * dCor(z, y)
    return cutA, z

def inferenceRepresentationHook(embeddingA, embeddingB, sampleIds, context, generator, self):
    return embeddingA, self.bottleneck(embeddingB)   # eval mode: deterministic
```
**Sweep.** bottleneck dim `k ∈ {8, 16, 32}` (vs `embeddingDim = 64`); `beta ∈ {1e-4, 1e-3, 1e-2}`;
`dcorStrength ∈ {0, 0.1, 1.0}`. The `k` sweep is the privacy dial: leak AUC/generalization is
bounded by the model's own test AUC **[VERIFIED-PAPER, Sun et al.]**, so a narrower `k` lowers
the ceiling for the attacker *and* for the main task.

**Attacks defended.** **Both.** This is the only shortlist entry whose defense argument is
information-theoretic rather than anchored to the auxiliary set, so it is robust to the attacker
changing its completion procedure.

**Honest tradeoff [INFERENCE].** The bottleneck is applied to `cutB` **at inference too**, so
unlike R1 it does not create a train/test mismatch — but it does permanently narrow the
information the top model can draw from B. Cost scales with `embeddingDim → k` compression and
with `dcorStrength`. Expect a smooth, tunable privacy/accuracy curve, with the same dataset
ordering as R1 (cheap on 03/05, dear on 02). MID's own claimed advantage is a better tradeoff
than fixed obfuscation; VMask's §I disputes that this family gets both, so **treat the
"free lunch" claim as unproven and tune conservatively.** **[VERIFIED-PAPER]**

### R3. Auxiliary-anchor inversion / decorrelation at train time, restored at inference
**Why it should work.** It attacks the attack's actual bottleneck: **10 labeled anchors per
class**. The completion head is a single `BatchNorm1d + Linear(64 → K)` fitting 20–30 points —
wildly overparameterized for that sample count — and the λU = 50 unlabeled term supplies *class
structure* while the anchors supply *class semantics*. Corrupt the anchor semantics and the
self-training can still cluster B's unlabeled features but cannot name the clusters.
**[VERIFIED-CODE + INFERENCE]** Supporting principle: [2603.18680](https://ar5iv.labs.arxiv.org/html/2603.18680)
shows disrupting the feature↔label alignment collapses LIA accuracy; and
[Zou et al., arXiv:2301.01142](https://ar5iv.labs.arxiv.org/html/2301.01142) describe MC as
inferring the label "by completing the local model with an additional layer and fine-tuning the
whole model using auxiliary labeled data". **[VERIFIED-PAPER]**

**Where.** `trainingRepresentationHook` (inject) + `inferenceRepresentationHook` (undo). The
public auxiliary CSV gives A the exact ID set, and `sampleIds` is delivered to both hooks.
**[VERIFIED-CODE]**

**Pseudocode.**
```
# precompute once: auxIds (set), auxLabel[id] (int)
def trainingRepresentationHook(cutA, cutB, labels, sampleIds, epoch, batchIndex,
                              context, generator, self):
    if self.anchorMode == "off" or labels is None:
        return cutA, cutB
    t = min(1.0, (epoch + 1) / self.anchorRampEpochs)     # ramp so early training is clean
    idx, flip = [], []
    for i, sid in enumerate(sampleIds):
        s = str(sid)
        if s in self.auxIds:
            idx.append(i)
            # label-inverting component for the FIRST class only  (half the anchors)
            flip.append(1.0 if self.auxLabel[s] == 0 else 0.0)
    if not idx:
        return cutA, cutB
    idx  = torch.tensor(idx,  device=cutB.device)
    flip = torch.tensor(flip, device=cutB.device, dtype=cutB.dtype).unsqueeze(1)
    out  = cutB.clone()
    # v = unit direction of the batch's between-class mean difference (label-relevant axis)
    with torch.no_grad():
        v = (cutB[labels == 1].mean(0) - cutB[labels == 0].mean(0))
        v = v / v.norm().clamp_min(1e-8)
    delta = 2.0 * self.anchorStrength * t * v            # push class-0 anchors past class-1 mean
    out[idx] = out[idx] + flip * delta                   # only class-0 anchors are moved
    return cutA, out

def inferenceRepresentationHook(embeddingA, embeddingB, sampleIds, context, generator, self):
    return embeddingA, embeddingB        # anchor correction is NOT applied at test time
```
**Attacks defended.** Primarily **PMC**; **AMC too**, because AMC only changes *how B trains*
— R3 operates on the alignment step that both PMC and AMC share.

**Honest tradeoff [INFERENCE].** This is the **lowest main-task cost** option in principle: the
manipulation touches only the ~20–30 auxiliary rows out of 7,738–43,489 (0.05–0.4% of data) and
only during joint training, and the inference hook restores the true geometry so the hidden test
path is unaffected. **Two real risks I want to be explicit about:**
1. **Train/test mismatch.** During training the top model sees class-0 anchors displaced by `δ`.
   If `δ` is large, the top model learns to rely on the displaced position, and at test time
   anchors appear undisplaced. Mitigations: ramp `δ` in late (after the top model has mostly
   converged), keep `δ` small relative to the between-class spread, or leave a fraction of
   class-0 anchors untouched as a consistency anchor.
2. **The attacker can compensate.** With 10 anchors per class, moving only 5 (or 10) of one class
   leaves the other class's anchors clean. A robust head — e.g. one that pools its 20 anchors,
   or the MixMatch EMA/BatchNorm machinery — may partially absorb the shift. The BatchNorm is
   the concerning part: it re-normalizes each batch, so an anisotropic shift can *increase*
   separability. **[INFERENCE]** This is exactly why R3 must be tuned, not assumed: it is the
   most *targeted* idea here and the most likely to interact non-monotonically with the attack's
   EMA/BatchNorm details. I could not test it.

Because of risk 2, I rank R3 third despite its attractive cost profile, and I would pair it with
R1 or R2 rather than ship it alone.

### R4. Label-DP / randomized-response label smoothing inside A's own loss
**Why it should work.** It attacks the same feature↔label alignment as R3, but *globally* rather
than at the anchors, and it is explicitly within the allowed scope ("A's loss function", "A's own
data, labels"). **[VERIFIED-PAPER]** ([Sun et al.](https://ar5iv.labs.arxiv.org/html/2203.01451)
describes randomized-response label flipping with a label-DP guarantee.)
**Where.** A's loss — reachable from the `trainingRepresentationHook`, which receives `labels`,
by writing the perturbed labels into the returned representation's context, or more simply by
registering a loss wrapper in A's training logic (`README.md:163-167` permits modifying A's own
loss function).
**Pseudocode.** For each batch, with probability `p` replace `y_i` with a uniform draw over the
other classes; train on the noisy labels; anneal `p` to 0 over the last ~20% of epochs so the
final model is fitted to clean labels.
**Attacks defended.** **Both**, since B's gradient is computed from the noisy loss and thus B's
features never fully bind to the true labels.
**Honest tradeoff.** **[VERIFIED-PAPER]** Sun et al. measured that label DP at ε = 0.5 **leaves
forward-embedding leakage at leak AUC ≈ 0.75** — i.e. label DP alone is *not sufficient* against
embedding-side attacks. It also directly corrupts B's learning signal, so the accuracy cost is
comparable to R1's but with less direct control. Ranked fourth for that reason: useful as a
complement, weak as a primary defense.

### R5. Class-mean gradient suppression — **read the warning before using**
The shipped `_centerClassConditional` (`defense/robust_vfl.py:343-360`) already implements the
natural form of this, and the repo's own note is that it "把 dataset_03 clean 打到 0.59" — it
**drops clean accuracy to 0.59**. **[VERIFIED-CODE]** I list it only because a *scheduled,
partial* version (suppress the class-mean component in the **last N epochs only**, at strength
≤ 0.3, ramped) has not been tried and is the most direct way to remove B's label-discriminative
subspace while leaving A's top model time to compensate. Expected outcome: real but modest PMC
reduction at a real accuracy cost. **Not recommended as a primary lever.**

---

## 6. Recommended combination and the honest bottom line

**Suggested configuration to evaluate first (in this order):**
1. **R2 at a conservative setting** (`k = 32` of 64, `beta = 1e-3`, `dcorStrength = 0.1`) as the
   *generalizable* defense — it is the only one whose guarantee does not depend on the attacker's
   completion procedure.
2. **R1 at `cutGradientScale ≈ 0.2`** with `rampEpochs = 30` as the *direct* lever on `θ_B`,
   exploiting AMC's [1,5] amplification cap.
3. **R3 as a tunable add-on**, validated for monotonicity before shipping.

**The tradeoff, stated without hedging.** Every mechanism available to A reduces *the amount of
label information B's embedding carries*, and the main task consumes that same information. There
is no free lunch available to A here — VMask buys its near-zero (0.09–0.34%) accuracy cost
precisely by paying in **protocol modification** (secret-sharing B's layers), which is the one
resource this competition forbids. So the achievable frontier under these hooks is strictly worse
than VMask's, and the practical task is to pick the point on that frontier per dataset. The
measured headroom table in §1.4 is the single most useful artifact for that decision: **on
dataset_03 and dataset_05 an aggressive defense is nearly free; on dataset_02 it is genuinely
expensive; dataset_01 and dataset_04 are intermediate.**

**What I could not do, and what that means for confidence.** No experiment was run: no PyTorch on
this machine, and no working outbound TLS from the shell. Therefore:
- The **hook-plumbing analysis (§1.2, §4)** is high confidence — it is a direct reading of the
  locked code.
- The **dataset-geometry table (§1.4)** is high confidence for what it measures (linear probes on
  raw features), but it is a *proxy* for the actual MLP bottom models in the competition. The
  qualitative ordering (dataset_03/05 cheap, dataset_02 expensive) is what I would bet on; the
  numbers are not the competition's numbers.
- The **ranking and expected effectiveness (§5)** is **[INFERENCE]**. R1's AMC-cap argument and
  R2's information-theoretic argument are the two I would defend most strongly. **R3 is the most
  likely to surprise in either direction.**
- No number in §3 attributed to another paper was cross-checked against a second source unless
  marked, and the items marked **[UNVERIFIED]** (Fu et al.'s own defense numbers, MID's numeric
  tables, CoAE's MC-specific numbers, LCD's mechanism, the VMask result table) should be treated
  as leads to verify, not facts.

---

## 7. Source list

**Primary attacks**
1. Fu, Zhang, Ji, Chen, Wu, Guo, Zhou, Liu, Wang. *Label Inference Attacks Against Vertical Federated Learning.* USENIX Security 2022. https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong
2. Liu, Zou, Kang, Liu, He, Yi, Yang. *Batch Label Inference and Replacement Attacks in Black-Boxed Vertical Federated Learning.* https://ar5iv.labs.arxiv.org/html/2112.05409
3. Sun, Yang, Yao, Wang. *Label Leakage and Protection from Forward Embedding in Vertical Federated Learning.* https://ar5iv.labs.arxiv.org/html/2203.01451
4. Li, Sun, Yang, Gao, Zhang, Xie, Smith, Wang. *Label Leakage and Protection in Two-party Split Learning.* ICLR 2022. https://ar5iv.labs.arxiv.org/html/2102.08504

**Defenses**
5. Tan, Zhang, Hu, Yang, Ran, Li. *VMask: Tunable Label Privacy Protection for VFL via Layer Masking.* NDSS 2025 / Front. Comput. Sci. 2027. https://ar5iv.labs.arxiv.org/html/2507.14629 · https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-51046-z
6. Zou, Liu, Zhang. *Mutual Information Regularization for Vertical Federated Learning.* https://ar5iv.labs.arxiv.org/html/2301.01142
7. Gu, Luo, Kang, Yao, Zhu, Li, Fan, Yang. *FedAdOb: Privacy-Preserving Federated Deep Learning with Adaptive Obfuscation.* https://ar5iv.labs.arxiv.org/html/2406.01085
8. Liu, Xu, Guo, Cao, Wang. *Revisiting Label Inference Attacks in VFL: Why They Are Vulnerable and How to Defend.* arXiv:2603.18680 (Mar 2026). https://arxiv.org/abs/2603.18680 · https://ar5iv.labs.arxiv.org/html/2603.18680
9. *Defend against Label Inference Attacks in VFL via Label Compression.* https://ieeexplore.ieee.org/abstract/document/11435639 · https://github.com/YuanShunJie1/LCD
10. Zou et al. *Defending Against Inference and Backdoor Attacks in VFL via Mutual Information Regularization.* https://xplorestaging.ieee.org/document/10825545

**Surveys / context**
11. Liu, Kang, Zou, Pu, He, Ye, Ouyang, Zhang, Yang. *Vertical Federated Learning: Concepts, Advances and Challenges.* https://ar5iv.labs.arxiv.org/html/2211.12814
12. *Label Leakage in Vertical Federated Learning: A Survey.* IJCAI 2024. https://www.ijcai.org/proceedings/2024/0902.pdf
13. Bai, Xu, Chen, Guo, Ji, et al. *VILLAIN: Backdoor Attacks Against Vertical Split Learning.* USENIX Security 2023. https://www.usenix.org/conference/usenixsecurity23/presentation/bai
14. *LFBA: Label-Free Backdoor Attacks in Vertical Federated Learning.* AAAI 2025. https://ojs.aaai.org/index.php/AAAI/article/view/34246
15. Huang et al. *A defense mechanism against label inference attacks in Vertical Federated Learning.* Neurocomputing 2025. https://www.sciencedirect.com/science/article/pii/S0925231225001481
16. *Vulnerabilities of label and model protection in VFL based on simulation models.* Neurocomputing 2025. https://www.sciencedirect.com/science/article/abs/pii/S0925231225025111
17. *GRAMSSAT: An efficient label inference attack against two-party split learning.* JISA 2025. https://dl.acm.org/doi/10.1016/j.jisa.2025.104159
18. *Privacy-Preserving Asynchronous VFL against Batch Level Label Inference Attacks.* ACM TOPS. https://dl.acm.org/doi/pdf/10.1145/3708282.3708295
19. *Attackers Are Not the Same! Unveiling the Impact of Feature Distribution on Label Inference Attacks.* https://ieeexplore.ieee.org/document/10752967
20. *Vertical federated learning: a structured literature review.* KAIS 2025. https://rd.springer.com/article/10.1007/s10115-025-02356-y

**This repository (locked protocol)**
21. `README.md` (permission boundary §"参赛权限边界", hook contract §"统一 A 方防御入口")
22. `core/trainer.py` — hook plumbing, `trainBatch`, AMC scaling
23. `attacks/label_inference/model_completion.py`, `active_model_completion.py`
24. `defense/robust_vfl.py` — shipped defense and its measured notes
25. `configs/dataset_0{1..5}.yaml`, `configs/scenarios/dataset_0{1..5}_pmc.yaml`, `configs/auxiliary_labels/`
