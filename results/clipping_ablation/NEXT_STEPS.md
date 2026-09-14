# 圖片已收集：下一步驗證獨立訓練模型

九張原始 PNG 與對照圖已加入 GitHub，見 [README](../../README.md#fixed-noise-preview-comparison)。不必再次收集或重跑這批圖片。

下一個研究問題是：換一份獨立訓練的模型，裁切造成的差異是否仍然存在？應先準備可設定訓練 seed、独立輸出資料夾與完整設定紀錄的訓練入口，再以相同 30,000 步預算訓練額外模型。僅修改原消融程式的 SEED 不會產生新模型。請勿直接以原資料夾呼叫 train()；它可能接續舊 checkpoint。

每份新模型先比較三種方式在 20、50、200 步的 FID，固定取樣 seed 作初步核對，再決定是否擴充取樣 seeds。報告時分開列出訓練與取樣變異。這些新訓練尚未執行，目前結論仍限於單一模型。

---

以下保留原始收集流程供追溯，不是目前需要重做的步驟。

# 下一步：收集已保存的圖片

不需要重跑訓練或 FID。在 Colab 掛載原本的 Google Drive 後，執行以下程式。
它選取與本研究 checkpoint 相符、seed=0 的完整實驗資料夾，
打包 3 種方式 × 20/50/200 步的九張 PNG，以及該資料夾的 results.json。
所有預覽在原實驗程式中使用固定 preview seed=2026，適合逐位置比較。

```python
from google.colab import drive, files
drive.mount('/content/drive')
from pathlib import Path
import json
import zipfile

root = Path('/content/drive/MyDrive/ddpm_mnist')
checkpoint_hash = 'c4145894a7dfd44a695396070fccd142c53fff3c06c46e4d04a66a4177b3adc6'
names = [f'{mode}_{steps}.png'
         for mode in ['no_clip', 'clip_original', 'clip_recompute']
         for steps in [20, 50, 200]]
candidates = []
for path in sorted(root.glob('clipping_ablation_*/results.json')):
    report = json.loads(path.read_text())
    config = report.get('config', {})
    if (config.get('seed') == 0
        and config.get('checkpoint_sha256') == checkpoint_hash
        and config.get('num_images') == 10000
        and config.get('batch_size') == 100
        and len(report.get('results', {})) == 12
        and all((path.parent / name).exists() for name in names)):
        candidates.append(path.parent)

if len(candidates) != 1:
    raise RuntimeError(
        f'找到 {len(candidates)} 個符合條件的資料夾：{candidates}。'
        '請檢查圖片是否齊全；若有多個版本，先確認要使用哪一份。'
    )
folder = candidates[0]
archive = Path('/content/clipping_ablation_previews.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in names + ['results.json']:
        z.write(folder / name, arcname=name)
print('來源：', folder)
files.download(str(archive))
```

將下載的 ZIP 上傳至對話，即可製作固定噪聲圖片比較。圖片應全部來自
同一資料夾，避免混用不同程式版本。不要將少數預覽視為 FID 的替代證據。

完成圖片整理後，下一個研究驗證是獨立訓練 checkpoint：
使用不同訓練 seeds、不同輸出資料夾與固定訓練预算；不得覆寫本次權重。
這需要另行準備訓練種子與 checkpoint 管理，不是只修改取樣 SEED。
