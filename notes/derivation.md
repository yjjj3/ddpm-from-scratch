# Derivation Notes

本文件整理 DDPM 的 ELBO、噪聲預測目標與 DDIM 的非馬可夫觀點，並對應本專案實作。這些是既有方法的推導，不構成 U 型 FID 曲線的成因證明。

## 1. Forward process 的封閉解與符號

令 $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s=1}^t\alpha_s$、$\bar\alpha_0=1$。數學時間為 $t=1,\ldots,T$，$x_0$ 為乾淨資料。

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I).
$$

反覆代入時，獨立高斯噪聲的變異數相加；由
$\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$，得到

$$
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
$$

這是 [Diffusion.q_sample](../ddpm_mnist.py) 一次取樣任意時間的依據，不必真的執行前面所有加噪步驟。

給定 $x_0$ 的條件變異數為 $(1-\bar\alpha_t)I$；若將資料也視為隨機變數，則

$$
\operatorname{Cov}(x_t)=\bar\alpha_t\operatorname{Cov}(x_0)+(1-\bar\alpha_t)I.
$$

只有原始協方差為 $I$ 時，無條件協方差才全程為 $I$。本程式的 Normalize(0.5, 0.5) 將像素映射到 $[-1,1]$，並不保證零均值或單位變異數。當 $\bar\alpha_T$ 足夠小，終點近似標準高斯；有限 $T$ 不保證精確相等。

## 2. ELBO 分解

### 2.1 為什麼引入 forward process？

我們想最大化 $\log p_\theta(x_0)$，但

$$
p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T},
\qquad
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t).
$$

高維潛變數與非線性神經網路轉移，使這個邊際似然通常無法有效地解析計算。選擇容易取樣且條件分布可計算的固定 forward process
$q(x_{1:T}\mid x_0)$ 作為變分分布，可以利用 Jensen 不等式：

$$
\begin{aligned}
\log p_\theta(x_0)
&=\log\mathbb E_q\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]\\
&\ge \mathbb E_q\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]
=:\operatorname{ELBO}(x_0)=-L(x_0).
\end{aligned}
$$

$q$ 並非模型的精確後驗；兩者差距恰好是
$\log p_\theta(x_0)-\operatorname{ELBO}(x_0)
=D_{\mathrm{KL}}(q(x_{1:T}\mid x_0)\|p_\theta(x_{1:T}\mid x_0))$。
DDPM 使用固定的 $q$，而非另外訓練一個 encoder。上述期望可用取樣估計，不代表整個 ELBO 已經是無需估計的數值。

### 2.2 貝氏定理與 telescoping

對 $t\ge2$，利用 Markov 性質與貝氏定理：

$$
q(x_t\mid x_{t-1})
=\frac{q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)}
{q(x_{t-1}\mid x_0)}.
$$

將 $t=1$ 單獨保留，避免使用退化的 $q(x_0\mid x_0)$：

$$
\begin{aligned}
q(x_{1:T}\mid x_0)
&=q(x_1\mid x_0)\prod_{t=2}^T
\frac{q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)}
{q(x_{t-1}\mid x_0)}\\
&=q(x_1\mid x_0)\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)}
\prod_{t=2}^Tq(x_{t-1}\mid x_t,x_0)\\
&=q(x_T\mid x_0)\prod_{t=2}^Tq(x_{t-1}\mid x_t,x_0).
\end{aligned}
$$

代回負 ELBO：

$$
L(x_0)=\mathbb E_q\left[
\log\frac{q(x_T\mid x_0)}{p(x_T)}
+\sum_{t=2}^T\log\frac{q(x_{t-1}\mid x_t,x_0)}
{p_\theta(x_{t-1}\mid x_t)}
-\log p_\theta(x_0\mid x_1)\right].
$$

先對各條件變數積分，即得

$$
L(x_0)=
\underbrace{D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))}_{L_T}
+\sum_{t=2}^T
\underbrace{\mathbb E_{q(x_t\mid x_0)}
D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t))}_{L_{t-1}}
+\underbrace{\mathbb E_{q(x_1\mid x_0)}[-\log p_\theta(x_0\mid x_1)]}_{L_0}.
$$

資料集目標再對 $x_0\sim q_{\mathrm{data}}$ 取期望。

| 項目 | 直觀意義 |
|---|---|
| $L_T$ | 終點加噪分布與生成先驗的差距；schedule 與先驗固定時不依賴 $\theta$。 |
| $L_{t-1}$ | 已知乾淨答案的去噪後驗，監督只看得到 $x_t$ 的模型轉移。 |
| $L_0$ | 最後一步的重建負對數似然。原 DDPM 影像模型使用離散化 decoder likelihood；本專案不另外計算此精確似然。 |

## 3. 從 KL 到 simplified loss

### 3.1 後驗的解析形式

對 $t\ge2$，由貝氏定理，後驗正比於兩個高斯的乘積：

$$
q(x_{t-1}\mid x_t,x_0)\propto
\exp\left[-\frac{\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2}{2\beta_t}
-\frac{\|x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0\|^2}
{2(1-\bar\alpha_{t-1})}\right].
$$

對 $x_{t-1}$ 配方，精度與均值為

$$
\tilde\beta_t^{-1}=\frac{\alpha_t}{\beta_t}+
\frac1{1-\bar\alpha_{t-1}},
\qquad
\tilde\mu_t=\tilde\beta_t
\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\right).
$$

化簡得到

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I),
$$

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
\qquad
\tilde\mu_t=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
$$

$t=1$ 時 $\tilde\beta_1=0$，不能直接套用以下非退化高斯 KL；它由 $L_0$ 單獨處理。

### 3.2 固定變異數的高斯 KL

設 $p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta,\sigma_t^2 I)$，
$\sigma_t^2>0$ 固定且不依賴 $\theta$。令影像維度為 $d$，則

$$
D_{\mathrm{KL}}(q\|p_\theta)=
\frac{\|\tilde\mu_t-\mu_\theta\|^2}{2\sigma_t^2}
+\frac d2\left(\frac{\tilde\beta_t}{\sigma_t^2}-1+
\log\frac{\sigma_t^2}{\tilde\beta_t}\right).
$$

第二項與 $\theta$ 無關，因此優化時只需均值距離。兩邊變異數不必相等；若相等，第二項為零。本專案的 DDPM sampler 選用 $\sigma_t^2=\beta_t$。

### 3.3 將均值預測重參數化為噪聲預測

由 forward 封閉解：

$$
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}{\sqrt{\bar\alpha_t}}.
$$

代入 $\tilde\mu_t$ 並合併 $x_t$ 的係數：

$$
\tilde\mu_t=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right).
$$

因此用網路預測 $\epsilon_\theta(x_t,t)$，並定義

$$
\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

兩個均值相減後，$x_t$ 項消失。對資料與噪聲取期望，得到

$$
\mathbb E_{x_0}L_{t-1}
=\mathbb E_{x_0,\epsilon}\left[
\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]+C_t,
$$

其中 $C_t$ 與 $\theta$ 無關。

### 3.4 為什麼改用 simplified loss？

Ho et al. 的 Eq. (14) 使用

$$
L_{\mathrm{simple}}=
\mathbb E_{\substack{t\sim\mathcal U\{1,\ldots,T\}\\
x_0\sim q_{\mathrm{data}},\,\epsilon\sim\mathcal N(0,I)}}
\left[\|\epsilon-\epsilon_\theta(
\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2\right].
$$

拿掉時間權重是選擇另一個 surrogate objective，不是與完整 ELBO 精確相等的代數化簡。原論文在其實驗中發現這樣的重加權有利樣本品質；相較其變分權重，簡化目標降低了小時間步的相對權重。這不表示小時間步無用，也不保證所有設定都改善或更適合似然估計。完整 ELBO 的 $L_0$ 亦不能直接等同這個 MSE。

[train() 的實作](../ddpm_mnist.py)：

~~~python
t = torch.randint(0, cfg.T, (x0.size(0),), device=device)
noise = torch.randn_like(x0)
xt = diffusion.q_sample(x0, t, noise)
pred = model(xt, t)
loss = F.mse_loss(pred, noise)
~~~

| 程式 | 數學對應 |
|---|---|
| torch.randint | 每張圖均勻抽取 timestep；程式索引 0 對應數學時間 1。 |
| torch.randn_like | 抽取標準高斯目標噪聲 $\epsilon$。 |
| diffusion.q_sample | 使用封閉解產生 $x_t$。 |
| model(xt, t) | 預測 $\epsilon_\theta(x_t,t)$。 |
| F.mse_loss(pred, noise) | 對 batch、通道與像素取平均的無時間權重 MSE；與每張圖平方範數的平均相差固定因子 $1/d$。 |

## 4. DDIM：non-Markovian 視角

### 4.1 同一組邊際，不同的聯合分布

$L_{\mathrm{simple}}$ 只需 $q(x_t\mid x_0)$ 與抽樣時間，不需要完整 forward 路徑。因此可構造具有相同條件邊際的另一族聯合分布，沿用噪聲預測網路。

這裡「非馬可夫」指 forward／inference process：轉移一般還依賴 $x_0$。生成端仍可以只依賴當前狀態及時間，並不需要保存整條歷史。標準 DDPM sampler 使用相鄰步轉移；加速取樣需要重新定義轉移，不能僅因 Markov 性質就宣稱任何方法都不能跳步。

### 4.2 構造並驗證 $q_\sigma$

對 $t\ge2$，令 $0\le\sigma_t^2\le1-\bar\alpha_{t-1}$，定義

$$
q_\sigma(x_{1:T}\mid x_0)=q(x_T\mid x_0)
\prod_{t=2}^Tq_\sigma(x_{t-1}\mid x_t,x_0),
$$

$$
q_\sigma(x_{t-1}\mid x_t,x_0)=
\mathcal N\left(
\sqrt{\bar\alpha_{t-1}}x_0+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}
\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},
\,\sigma_t^2I\right).
$$

其構造方式是：若已知 $x_t$ 的條件邊際正確，則
$u_t=(x_t-\sqrt{\bar\alpha_t}x_0)/\sqrt{1-\bar\alpha_t}\sim\mathcal N(0,I)$。
取與 $u_t$ 獨立的 $z\sim\mathcal N(0,I)$，令

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+
\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}u_t+\sigma_tz.
$$

給定 $x_0$，均值為 $\sqrt{\bar\alpha_{t-1}}x_0$，變異數為
$(1-\bar\alpha_{t-1}-\sigma_t^2)I+\sigma_t^2I=(1-\bar\alpha_{t-1})I$。
從終點向後歸納，即得到所有所需邊際。

DDIM 論文將正 $\sigma$ 的變分目標寫成加權噪聲目標加常數。不同權重共享最優解的論證有跨時間不共享參數等理想化條件；本專案 U-Net 共享參數，因此沿用網路是 surrogate 訓練下的實務做法，不能宣稱對所有有限模型有完全相同的最優解。$\sigma=0$ 是退化的確定性極限，不直接落在正變異數定理的條件內。

### 4.3 跳步、確定性與 eta

取時間子序列 $0=\tau_0<\tau_1<\cdots<\tau_S=T$。以 $t=\tau_i$、
$s=\tau_{i-1}$ 表示一般跳步；在選定時間上用同樣的邊際構造短鏈，即可沿用網路，不需重新訓練。

先估計乾淨影像：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

再使用

$$
\sigma_{t\to s}=\eta
\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}}
\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}},
$$

$$
x_s=\sqrt{\bar\alpha_s}\hat x_0+
\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\epsilon_\theta(x_t,t)
+\sigma_{t\to s}z,\qquad z\sim\mathcal N(0,I).
$$

通常取 $0\le\eta\le1$，確保此設定下根號合法。$\eta$ 線性縮放標準差，變異數隨 $\eta^2$ 改變。

當 $\eta=0$：

$$
x_s=\sqrt{\bar\alpha_s}\hat x_0+
\sqrt{1-\bar\alpha_s}\epsilon_\theta(x_t,t).
$$

固定模型、時間子序列與初始 $x_T$ 後，更新不新增隨機噪聲；初始 $x_T$ 仍隨機抽取。允許跳步不表示不同步數輸出相同，也不保證任意短鏈保持品質；學到的近似網路並非已知 $x_0$ 的真實條件分布。

當 $\eta=1$ 且 $s=t-1$，有 $\sigma_{t\to s}^2=\tilde\beta_t$，對應使用後驗變異數的 DDPM 形式。本專案 p_sample_loop 使用 $\beta_t$，且沒有同樣的逐步 $\hat x_0$ 裁切，所以不能說 eta=1 與本專案 DDPM sampler 完全相同。跳步時應使用上述一般式，不是原始相鄰步的 $\tilde\beta_t$。

### 4.4 與 ddim.py 逐行對應

以下對照 [ddim_sample()](../ddim.py) 的程式敘述；使用函式與片段定位，避免新增文字後行號失效。

| 程式片段 | 意義 |
|---|---|
| ts = torch.linspace(...) | 選取遞減取樣時間；num_steps 控制網路呼叫次數。 |
| x = torch.randn(shape, device=device) | 初始高斯噪聲 $x_T$。 |
| t_prev = ... else -1 | 下一個子序列時間；-1 是乾淨終點的特殊標記。 |
| ab_t = diffusion.alpha_bars[t] | 程式索引 $t$ 對應數學 $\bar\alpha_{t+1}$。 |
| ab_prev = ... else torch.tensor(1.0, ...) | 下一時間的累積係數；乾淨終點使用數學 $\bar\alpha_0=1$。 |
| eps = model(x, t_batch) | 噪聲估計 $\epsilon_\theta$。 |
| x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt() | 未裁切的 $\hat x_0$。 |
| x0_pred = x0_pred.clamp(-1, 1) | 額外的裁切啟發式，不是上面未裁切公式直接推出的操作。 |
| sigma = eta * (...) | 一般跳步的 $\sigma_{t\to s}$。 |
| dir_xt = (1 - ab_prev - sigma**2).sqrt() * eps | 保留的噪聲方向項。 |
| x = ab_prev.sqrt() * x0_pred + dir_xt | 合成訊號估計與方向項。 |
| if eta > 0 and t_prev >= 0: x = x + sigma * torch.randn_like(x) | 非終點且 eta>0 時加入新噪聲；終點的 sigma 為零。 |

裁切後，程式仍沿用原先的 eps；裁切過的 x0_pred 與 eps 未必再滿足原本的 $x_t$ 分解。這是理論公式與現有實作的差異，不應隱藏，也不能未經實驗就認定它造成 U 型 FID。本次文件更新不改變 sampler 行為。

## 5. 後續驗證：U 型 FID 曲線

目前僅能陳述：在本次模型與已測試設定中，20 步取得最低 FID。成因與泛化性尚待驗證。下列項目是計畫，尚未完成：

1. 固定模型與起始噪聲，比較不裁切、目前裁切、裁切後重算一致 eps 的消融實驗。
2. 使用獨立訓練種子得到多個 checkpoint；分開報告訓練變異與取樣變異。
3. 補測 15、25、30 等鄰近步數及完整步數基準；用驗證設定選擇步數，再做最終測試。
4. 加入固定噪聲圖片對照、領域相關品質與多樣性指標，以及固定硬體下排除存檔和 FID 計算的生成時間。
5. 擴展 eta、noise schedule 與資料集；保存 checkpoint 識別、設定、各 seed 原始分數，再重畫平均值與誤差棒一致的圖。

## References

- Ho, Jain & Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Eqs. (4)–(14), Appendix A.
- Song, Meng & Ermon (ICLR 2021), [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), Sections 3–4 and appendices. 該文的 $\alpha_t$ 表示累積係數，對應本文件的 $\bar\alpha_t$。
