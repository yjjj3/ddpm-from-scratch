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
