# Research Summary：DDIM 裁切如何影響步數與生成品質？

## 問題與方法

本專案以從零實作的 PyTorch DDPM／MNIST 為基礎，研究一個具體問題：為何增加 DDIM 取樣步數，FID 反而變差？比較三種更新：不做中間裁切（no_clip）、裁切乾淨影像估計但沿用原噪聲（clip_original）、裁切後重算一致噪聲（clip_recompute）。固定 checkpoint、評估協定與每個 seed 的初始噪聲序列，避免將重新訓練造成的差異混入同模型比較。

證據基準為 commit `3115979ea808b9133a0c61a5bff8f56f579b40cf` 的六份原始 JSON：原 checkpoint 36 筆，新增 checkpoint 27 筆，共 63 筆；共同 20／50／200 步網格有 54 筆。每個 checkpoint 使用取樣 seeds 0、42、123，每筆生成 10,000 張影像，eta=0，對 padded MNIST test images 計算 clean-mode FID。所有模式輸出前均裁切至 [-1,1]。本次只重算存檔統計與局部代數，未重新生成或計算 FID。

## 具體結果與未符合預期的結果

下列是**既有資料的事後探索性分析**；不是事先註冊的假設驗證。各 checkpoint 分開統計，不將六個取樣結果當成六次独立訓練。

| 比較 | 原 checkpoint | 新 checkpoint（設定記錄 training seed=2026） |
|---|---:|---:|
| 200 步 clip_original 平均 FID | 17.471 | 21.481 |
| 200 步 clip_recompute 平均 FID | 5.680 | 5.419 |
| no_clip 的平均 FID(200)−FID(50) | +0.569 | +0.444 |
| clip_recompute 的平均 FID(200)−FID(50) | +0.559 | +0.416 |
| 50 步 clip_recompute−no_clip 平均 FID | +0.243 | +0.240 |

重算噪聲大幅縮小了相對原裁切方式的 200 步 FID，但**沒有恢復隨步數增加而單調改善**：兩種替代模式在每個 checkpoint 的全部三個取樣 seeds，200 步都比 50 步差。共同 seed×step 條件下，no_clip 在兩個 checkpoint 合計 18/18 個配對的 FID 都低於 clip_recompute；這些配對共享模型與生成設定，不能當作 18 次獨立訓練或母群顯著性證據。

目前已測網格的最低 FID：原裁切模式為 20 步，其餘模式為 50 步。這是以測試集選出的探索性最低點，並非保證最優步數。恢復局部代數一致性不等於最佳 FID，也不足以解釋完整非單調現象。

## 研究意義、限制與下一步

研究價值在於把「步數越多越好」拆成可檢查的實作假設，保留改善與未改善的部分，而非提出新模型或宣稱新機制。裁切與噪聲方向的一致性可由公式和 CPU 合成值驗證；FID 差距和高頻誤差累積之間的因果關係尚未建立。

兩個不同 checkpoint hash、新模型的 30,000 completed steps／EMA metadata 與訓練設定提供有限來源證據，**不獨立證明完整、不中斷或彼此獨立的訓練歷程**。原模型訓練 seed 未在這批 JSON 記錄。各模型內 sample SD 只描述三個取樣 seeds 的變異；不涵蓋訓練變異、真實參考集重抽樣或 FID 偏誤。歷史 clean-fid 精確版本未記錄；preview 僅屬原模型；完整 FID 生成影像與逐步軌跡未隨這批資料保存。

下一個有判別力的工作是先在固定 incoming state 上比較三種單步更新，量測裁切幅度與分解殘差，再規劃配對軌跡診斷；不能從終點超界比例直接推導高頻機制。任何新增生成、FID、訓練或跨資料集實驗均需另行同意。具體反證條件與資源範圍見[假設與驗證計畫](research_hypotheses.md)。

## 證據與重現

- [README：完整協定、原始 JSON 與 checkpoint 來源](../README.md)
- [配對差值與原始檔 SHA-256](../results/clipping_ablation/evidence_audit.md)；[逐 seed CSV](../results/clipping_ablation/paired_differences.csv)
- [代數推導與假設](research_hypotheses.md)；[DDPM／DDIM 推導](derivation.md)
- 在 repo 根目錄執行 `python3 scripts/audit_research.py`，只需 Python 標準函式庫；檢查 63 筆來源、配對差值及 60 個合成標量情境。此檢查不等同執行 notebook／模型或重現 FID。
