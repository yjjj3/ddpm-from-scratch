# 可證偽假設、負面結果與驗證邊界

本文件區分**事後整理的可檢查命題**、**代數恆等式**與**尚未執行的未來假設**。現有數據已被查看，不聲稱預註冊；以下符號不引入新的實驗結果。

## 1. 現有資料能回答什麼？

| 事後命題 | 反證／檢查方式 | 存檔結果與結論 |
|---|---|---|
| 200 步重算噪聲的 FID 低於原裁切方式 | 在同 checkpoint、seed、step 比較；任一反向配對反駁「所有已測配對都改善」 | 兩模型各 3/3 改善；平均差 original−recompute 為 11.791／16.062。只支持這些設定。 |
| 移除分解不一致後，增加步數就不會惡化 | 檢查每個 seed 的 FID(200)−FID(50) 是否 ≤0 | no_clip 與 recompute 均在兩模型各 3/3 >0。存檔反駁這個在已測點成立的強命題，沒有估計母群期望。 |
| 代數一致的 clip_recompute 也有最低 FID | 與同 seed、step 的 no_clip 比較 | 共同網格 18/18 配對反向；no_clip 較低。重算不是目前最佳 FID 的充分條件。 |
| 原裁切處理足以解釋全部非單調性 | 消除該不一致後若仍回升，便不支持充分解釋 | 兩替代模式仍回升；剩餘成因未知。不能直接指定為高頻誤差或離散化誤差。 |

來源與每個配對的數字見[自動稽核表](../results/clipping_ablation/evidence_audit.md)及[CSV](../results/clipping_ablation/paired_differences.csv)。不以多個相依對比進行事後顯著性宣稱。原 notebook 的 checkpoint guard RuntimeError 是保護性中止，不是模型訓練失敗證據。

## 2. DDIM clipping 的局部代數

對固定 incoming state $x_t$ 與固定網路預測 $\epsilon$，令 $a=\bar\alpha_t$、$b=\bar\alpha_s$，$0<a<1$、$a\le b\le1$，eta=0，並定義

$$
\hat x_0=\frac{x_t-\sqrt{1-a}\epsilon}{\sqrt a},\quad
c=\operatorname{clip}(\hat x_0,-1,1),\quad \delta=c-\hat x_0.
$$

沿用原噪聲時，重建值減去 incoming state 的殘差為

$$
\sqrt a c+\sqrt{1-a}\epsilon-x_t=\sqrt a\delta.
$$

重算噪聲則為

$$
\epsilon'=\frac{x_t-\sqrt a c}{\sqrt{1-a}}
=\epsilon-\sqrt{\frac a{1-a}}\delta,
$$

使 $\sqrt a c+\sqrt{1-a}\epsilon'=x_t$。這是代數一致性，不代表 $c$ 是真實乾淨影像或 $\epsilon'$ 是真實噪聲。

相對 no_clip 的**同一狀態單步更新差**：

$$
x_s^{\mathrm{original}}-x_s^{\mathrm{no\_clip}}=\sqrt b\delta,
$$

$$
x_s^{\mathrm{recompute}}-x_s^{\mathrm{no\_clip}}
=\left[\sqrt b-\sqrt{\frac{(1-b)a}{1-a}}\right]\delta.
$$

可檢查的邊界：未觸發裁切時三者相同；零時間跨度 $b=a$ 時 no_clip/recompute 回到 $x_t$，original 若裁切啟動仍改變狀態；乾淨終點 $b=1$ 時方向項為零，original/recompute 都輸出 $c$。零跨度只是假想診斷，現有 sampler 不使用重複 timestep。

對應 [review notebook](../ddpm_experiments_review.ipynb) 的 `run_ablation` → `sample`：先算 x0、視 mode clamp，僅 `clip_recompute` 以 `(x - sqrt(ab)*x0)/sqrt(1-ab)` 更新 eps，再合成下一步。[ddim.py](../ddim.py) 仍只有 original clipping。三種模式沿軌跡會遇到不同狀態、模型預測與 delta，故不能把上述固定狀態公式直接累加成 FID 因果證明。

`python3 scripts/audit_research.py` 用標準函式庫檢查 60 個合成標量情境（含 active/inactive clipping、零跨度及終點）；`math.isclose` 使用絕對容差 1e-12、相對容差 1e-9（Python 預設值）；程式中的邊界相等檢查另採精確相等。它驗證公式，沒有執行 PyTorch sampler。小 $a$ 或 $1-a$ 的實際精度、裝置與 dtype 仍需獨立實作測試。

## 3. 尚未執行、需先同意的判別性計畫

| 未來假設 | 預先指定的檢查與反證 | 最小資源／限制 |
|---|---|---|
| 實際三模式程式符合上述局部代數 | 固定模型、同一 incoming state 與 eps，比對公式殘差；float32 先以 atol=1e-5、rtol=1e-4 檢查並記錄誤差，不因失敗放寬容忍度來掩蓋問題 | CPU 小張量與 toy predictor 即可先做實作檢查；真 checkpoint 需先取得並驗 hash，不需訓練或 FID。 |
| 大步數惡化伴隨可量測的裁切／頻域變化 | 固定一組 latents，於預定 timestep 記錄 clip fraction、幅度、分解殘差、背景高頻能量；若特徵不隨回升出現，便削弱此關聯假設。即使相關仍需干預才能討論機制 | 每 checkpoint 16 latents × 3 modes × (50+200) steps = 12,000 sample-level network evaluations；两個模型 24,000。只保存彙總與少量預覽，無 FID。牆鐘與記憶體須小規模試測，不能由 FID 表保證。 |
| 50 步附近優勢在未參與選擇資料仍成立 | 預先固定驗證／測試參考資料與步數候選；驗證集選步數後只做一次持出評估，保留反向結果 | 例如 3 modes × 5 step counts × 3 seeds × 10k images = 每 checkpoint 45 次／450k 張生成，另加最終評估；目前不執行。不得把新協定與舊 test-selected 數字混合。 |
| 趨勢不限於這兩個 checkpoint／MNIST 協定 | 先補可驗證訓練歷程，再預定獨立 seeds 或單一替代資料集／schedule；若配對效果反向則限縮結論 | 新獨立 run 若沿用設定需 30k optimizer steps、batch 128；GPU 時間、儲存與費用尚未估計。跨資料集另定 preprocessing／metric，不能以原 MNIST FID 直接比較。 |

優先順序為免費存檔分析、實作一致性、有限診斷，最後才是新生成／評估／訓練。現存 JSON 的終點超界比例只有比例，沒有幅度；clip 模式的零比例多由終點運算保證。缺少 FID 全量影像與逐步軌跡，因此目前無法補算頻域機制證據。本次不啟用 GPU、付費資源或任何新實驗。
