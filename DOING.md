# ASEE Performance Optimization (GPU Offloading & High-Speed Pipeline)

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
