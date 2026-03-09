# ASEE Performance Optimization (GPU Offloading & High-Speed Pipeline)

---

## 🔬 調査中: GpuYuNetDetector `_postprocess` のデコード公式バグ

### 症状
- `--detection-backend onnxruntime` 使用時、1人の人物に対して18件など大量の誤検出が発生
- `cv2.FaceDetectorYN`（デフォルト backend）では正常に1件検出

### 誤った仮説（棄却済み）
- ~~入力を `/255.0` で正規化すべき~~
  → YuNet ONNX モデルは `[0, 255]` の生ピクセル値を期待。`/255.0` は検出ゼロになる副作用を引き起こす

### 根本原因（特定済み、未修正）

`_postprocess` の **バウンディングボックスデコード公式が2点間違っている**。

#### 1. アンカー中心オフセットの誤り
```python
# 現在（誤）:
cx = (x_idx + 0.5) * stride
cy = (y_idx + 0.5) * stride

# 正しい:
cx = x_idx * stride
cy = y_idx * stride
```

#### 2. bbox エンコード形式の誤り
```python
# 現在（誤）: FCOS [l, t, r, b] として直線デコード
x1 = (cx - bbox[:, 0] * stride) * scale_x
x2 = (cx + bbox[:, 2] * stride) * scale_x
w_box = x2 - x1

# 正しい: [dx, dy, log_w, log_h] 形式（NanoDet スタイル）
xc = (cx + bbox[:, 0] * stride) * scale_x   # 中心x = anchor_cx + dx*stride
yc = (cy + bbox[:, 1] * stride) * scale_y   # 中心y = anchor_cy + dy*stride
w_box = np.exp(bbox[:, 2]) * stride * scale_x  # 幅 = exp(log_w) * stride
h_box = np.exp(bbox[:, 3]) * stride * scale_y  # 高さ = exp(log_h) * stride
x1 = xc - w_box / 2
y1 = yc - h_box / 2
```

キーポイントのデコードも anchor center を `x_idx * stride`（オフセットなし）に変更する必要あり。

### 検証
`/tmp/test_face.jpg`（Lena 画像を 640×640 にリサイズ）で実測：
- 現行コード：8件検出（小さいボックスが分散、NMS で抑制されない）
- `cv2.FaceDetectorYN`：1件検出、box=`[261.6, 227.2, 177.8, 257.1]`
- 修正後公式（実験済み）：1件検出、box=`[261.6, 227.2, 177.8, 257.1]`（完全一致）

### 次のアクション
1. `gpu_yunet.py` の `_postprocess` を上記正しい公式に書き直す
2. キーポイントデコードも anchor center 修正に合わせて更新
3. 既存テストを修正・新規リグレッションテスト追加
4. サーバー再起動して `peopleCount` が正常値（1人=1件×カメラ台数程度）になることを確認

---



## 目的
RTX 3060 x 2 と Ryzen 9 5950X という強力なハードウェアを最大限に活かし、現在の CPU 負荷（236%超）を劇的に削減しつつ、リアルタイム 60 FPS の滑らかな顔認識を実現する。

## これまでの進捗

### Wave 1: 推論エンジンの GPU 化 (Completed)
- 従来の `cv2.FaceDetectorYN` (OpenCV DNN / CPU) から `onnxruntime-gpu` を使用した **`GpuYuNetDetector`** へ移行。
- RTX 3060 上での CUDA 加速を有効化し、1カメラあたりの推論速度を大幅に向上させた。

### Wave 2: プリプロセッシングの GPU 加速 (Completed)
- **PyTorch 統合**: CPU で行っていた映像のリサイズと正規化を PyTorch (`torch.nn.functional.interpolate`) を用いて GPU 上で完結させるよう実装。
- **IO Binding**: GPU メモリ上の Tensor ポインタを直接 ONNX Runtime に渡すゼロコピー設計を採用し、CPU-GPU 間の無駄な転送を排除。

### Wave 3: 配信・描画の最適化 (Completed)
- **グリッド・プリレンダリング**: 毎フレーム CPU で計算していた HUD のグリッド描画を、初期化時に一度だけ生成して使い回す方式に変更。
- **PyTurboJPEG 導入**: MJPEG 配信用の JPEG 圧縮を、標準 libjpeg から SIMD 最適化された `libjpeg-turbo` へ差し替え。CPU 負荷を約 5% 削減。

### Wave 4: 集中ワーカーとバッチ推論の導入 (In Progress)
- **アーキテクチャ刷新**: 4台のカメラごとに独立していた推論スレッドを廃止し、唯一の集中ワーカー (`_face_detect_worker_centralized`) が全カメラを統括する設計へ変更。
- **バッチ推論の実装**: モデルの制約により完全な一括処理（Batch Size 4）は断念したが、単一ワーカーによる「高速な順次 GPU 処理」によりスレッド間競合を排除。

### Wave 4: 集中ワーカーとバッチ推論の導入 (Completed)

- **Reshape Error 修正**: `_postprocess` 内の特徴マップ計算を `_target_w/_target_h`（640×640）に統一。`_input_w/_input_h`（キャプチャ解像度）を使っていた誤りを修正。
- **座標スケーリング追加**: 検出座標をモデル出力空間（640×640）からキャプチャ空間（1280×720）へ `scale_x/scale_y` でスケールバック。
- **`FaceBox.from_yunet_row()` 追加**: 集中ワーカーで使う GpuYuNetDetector の ndarray 出力を FaceBox へ変換するクラスメソッドを実装。
- **turbojpeg optional 化**: `turbojpeg` が未インストールでも OpenCV フォールバックでパッケージ全体がロード可能に。

## 現在の状態
- 全 178 テスト PASS（回帰テスト含む）
- mypy: 変更起因のエラーなし
- 1280×720 キャプチャ + 640×640 モデルの組み合わせが正常動作

## 次のステップ
1. 実機 (`--allow-live-camera --capture-profile 720p`) での 60 FPS 安定稼働を確認。
2. プロダクション環境への正式マージ（main へ push 済み）。

---
*Created by YuiClaw (Gemini) on 2026-03-10*
