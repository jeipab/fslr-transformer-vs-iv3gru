# **MASTER LIST OF ALL FORMULAS USED IN THE STATISTICAL ANALYSIS PIPELINE**

_(Fully aligned with `statistical_analysis.py`)_

---

# **I. Data Cleaning and Pairing**

### **1. Remove NaN pairs**

The script removes any row where _either value_ is NaN:

$$
X_i = \text{Transformer}[i],\quad
Y_i = \text{GRU}[i]
$$

$$
\text{valid\_mask}_i = \neg(\text{isnan}(X_i) \lor \text{isnan}(Y_i))
$$

$$
X_i \leftarrow X_i[\text{valid\_mask}],\quad
Y_i \leftarrow Y_i[\text{valid\_mask}]
$$

### **2. Compute paired differences**

$$
d_i = X_i - Y_i
$$

### **3. Remove zero-differences for Wilcoxon**

(Scipy's `zero_method='wilcox'`)

$$
d^{*} = {d_i \mid d_i \neq 0}
$$

$$
N = |d|,\qquad N_{\text{nonzero}} = |d^{*}|
$$

---

# **II. Descriptive Statistics**

### **4. Mean of each model**

$$
\texttt{transformer\_mean} = \frac{1}{N}\sum X_i
$$

$$
\texttt{gru\_mean} = \frac{1}{N}\sum Y_i
$$

### **5. Mean difference**

$$
\texttt{difference} = \bar{d} = \frac{1}{N}\sum d_i
$$

### **6. Standard deviation of differences**

(Used for t-test and Cohen's d calculation; computed with `ddof=1` for sample standard deviation)

$$
s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(d_i - \bar{d})^2}
$$

This matches:

```python
s_d = diff.std(ddof=1)
```

### **6b. Variance check**

(Performed before statistical testing)

$$
\text{has\_variance} = \neg(\text{all}(d_i = 0) \lor (N > 1 \land s_d = 0))
$$

If no variance exists, statistical tests are not applicable.

---

# **III. Normality Testing — Shapiro–Wilk**

(applied to `diff`)

### **7. Shapiro–Wilk statistic**

(Computed by Scipy's `shapiro` function)

$$
W = \frac{\left( \sum_{i=1}^{n} a_i d_{(i)} \right)^2}{\sum_{i=1}^{n}(d_i - \bar{d})^2}
$$

where $d_{(i)}$ are the ordered differences and $a_i$ are coefficients derived from the expected values of order statistics of a standard normal distribution. The test is only performed when $N \geq 3$ and variance exists.

### **8. Normality decision rule**

$$
p_{\text{SW}} > 0.05 \text{ and } N \geq 2 \text{ and variance exists} \Rightarrow \text{Use paired t-test}
$$

$$
p_{\text{SW}} \le 0.05 \text{ and } N_{\text{nonzero}} \geq 2 \text{ and variance exists} \Rightarrow \text{Use Wilcoxon}
$$

**Note**: If $N < 3$ or no variance exists, normality test is skipped and test selection depends on other conditions.

---

# **IV. Paired t-test**

(Only used if data is normal, $N \geq 2$, and variance exists)

**Conditions**: Normal distribution (Shapiro–Wilk $p > 0.05$), $N \geq 2$, and non-zero variance in differences.

### **9. t statistic**

(Computed by Scipy's `ttest_rel` function)

$$
t = \frac{\bar{d}}{s_d/\sqrt{N}} = \frac{\bar{d}\sqrt{N}}{s_d}
$$

### **10. Degrees of freedom**

$$
df = N - 1
$$

### **11. p-value**

$$
p = 2\left(1 - F_t(|t|; df)\right)
$$

### **12. Cohen's d**

(Paired samples Cohen's d, computed exactly as in code)

$$
d_{\text{cohen}} = \frac{\bar{d}}{s_d} = \frac{\text{mean}(d_i)}{\text{std}(d_i, \text{ddof}=1)}
$$

This matches:

```python
cohens_d = diff.mean() / diff.std(ddof=1)
```

**Note**: Returns `NaN` if standard deviation is zero (no variance).

---

# **V. Wilcoxon Signed-Rank Test**

(Used when data is non-normal, $N_{\text{nonzero}} \geq 2$, and variance exists)

**Conditions**: Non-normal distribution (Shapiro–Wilk $p \leq 0.05$) or normality test not applicable, $N_{\text{nonzero}} \geq 2$, and non-zero variance in differences.

### **13. Rank absolute non-zero differences**

$$
R_i = \operatorname{rank}(|d^{*}_i|)
$$

### **14. Positive and negative rank sums**

$$
S^+ = \sum_{d^{*}_i > 0} R_i
$$

$$
S^- = \sum_{d^{*}_i < 0} R_i
$$

### **15. W statistic used by Scipy**

(Scipy's `wilcoxon` function with `zero_method='wilcox'` and `alternative='two-sided'`)

Scipy returns the _smaller_ of the positive/negative rank sums:

$$
W = \min(S^+, S^-)
$$

This matches:

```python
w_stat, p_val = wilcoxon(transformer, gru, zero_method='wilcox', alternative='two-sided')
```

---

# **VI. Wilcoxon Z-Value**

(Computed explicitly by the script using `calculate_wilcoxon_z`)
**IMPORTANT:** The script uses **non-zero pairs** ($N_{\text{nonzero}}$), NOT total N.

### **16. Expected value of W**

$$
\mu_W = \frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)}{4}
$$

### **17. Standard deviation of W**

$$
\sigma_W =
\sqrt{
\frac{N_{\text{nonzero}}(N_{\text{nonzero}}+1)(2N_{\text{nonzero}}+1)}{24}
}
$$

### **18. Z-value computed by the script**

$$
z = \frac{W - \mu_W}{\sigma_W}
$$

This matches:

```python
z = calculate_wilcoxon_z(w_stat, n_nonzero)
```

---

# **VII. P-value for Wilcoxon (large N)**

(When `n_nonzero >= 10`)

### **19. Two-tailed p-value**

(Computed by Scipy for large $N_{\text{nonzero}} \geq 10$)

$$
p = 2\left(1 - \Phi(|z|)\right)
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

---

# **VIII. Effect Size for Wilcoxon (r)**

(Only when `n_nonzero >= 10`)

### **20. Effect size formula**

$$
r = \frac{|z|}{\sqrt{N_{\text{nonzero}}}}
$$

This matches:

```python
effect_size = abs(z) / sqrt(n_nonzero)
```

---

# **IX. Exact Wilcoxon Test (small N)**

(When `n_nonzero < 10`)

### **21. Exact statistic**

$$
W = \min(S^+, S^-)
$$

### **22. p-value**

(Computed by Scipy's exact distribution for small $N_{\text{nonzero}} < 10$)

No closed-form formula; obtained from Scipy's exact permutation distribution of the Wilcoxon W statistic:

$$
p = P(W \leq w_{\text{obs}} \mid H_0)
$$

where $w_{\text{obs}}$ is the observed W statistic and the probability is computed from the exact null distribution.

### **23. Effect size**

Not computed:

$$
r = \text{N/A}
$$

---

# **X. Direction and Decision Rules**

### **24. Direction**

$$
\text{If } \bar{d} > 0 \Rightarrow \text{Transformer > IV3-GRU}
$$

$$
\text{If } \bar{d} < 0 \Rightarrow \text{IV3-GRU > Transformer}
$$

### **25. Hypothesis decision**

(Alpha level: $\alpha = 0.05$)

$$
p < 0.05 \Rightarrow \text{Reject Null Hypothesis}
$$

$$
p \ge 0.05 \Rightarrow \text{Fail to Reject Null Hypothesis}
$$

**Note**: In the output, hypotheses are labeled as "Null Hypothesis 1" (for recognition) or "Null Hypothesis 2" (for classification). If no variance exists, the decision is always "Fail to Reject Null Hypothesis".

---
