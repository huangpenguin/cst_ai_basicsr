# Vertex AI デプロイ・検証結果（上司確認用）

2026-11-18 ソース整理：本書の未コミット・未pushの記述は検証当時の状態です。ソースと運用手順はGitで管理し、実行記録はNFSの `/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/` に保存します。クラウドの再デプロイ・予測はこの整理では実施していません。

2026年11月17日の検証・確認記録に基づく要約です。

## 1. 構成と利用方法

**社内 Gradio 画面で画像を受け付け、推論は東京の Vertex AI GPU で実行します。** 社内サーバーは画面・アップロード・ダウンロードを担当し、ローカル GPU は使用しません。

| 項目 | 内容 |
|---|---|
| 利用 URL | http://192.168.3.18:7860 |
| GCP プロジェクト | `hallowed-pipe-497802-i0` |
| クラウド構成 | 東京 `asia-northeast1`、`n1-standard-8` + T4 ×1 |
| 推論モデル | 既存のシリンダーヘッド用 final（30,000 step） |
| 自動起動・停止 | 0～1 副本。無通信時の待機設定は300秒、実測では処理終了後約6分で0に縮小 |

社内ネットワークから URL を開き、画像をアップロードして「Denoise」を押すと、結果を16 bit TIFFでダウンロードできます。休止後の初回は起動待ちがあるため、画面を閉じずに待ちます。別の物理 PC からの接続確認は未実施です。

2026年11月18日追記：画面を英語に戻し、元の11モデルの選択一覧を復元しました。現在のVertex推論で使えるのはCylinderhead finalのみで、他のモデルは未デプロイと表示し、実行を無効にしています。work1への移行は、依頼者の通知を待って実施します。

画面は社内限定ですが、Vertex との通信は IAM 認証付きの公開 HTTPS API です。VPN による完全な私設接続ではありません。画面独自のログイン機能はなく、バックエンドの認証は現在の Huang のユーザー資格情報を使用します。

## 2. 検証結果

| 検証 | 結果・実測値 |
|---|---|
| 小画像の推論 | 64×64 PNG：クラウド処理0.777秒、送信～ダウンロード6.809秒 |
| 代表画像の推論 | 1518×1518・16 bit TIFF：クラウド処理46.364秒、全体61.912秒 |
| 休止後の自動起動 | 64×64 PNG：全体200.293秒（約3分20秒）、クラウド処理1.635秒 |
| 空き GPU の解放 | 起動後の処理終了から366.712秒で副本0を確認 |
| 学習 | Vertex T4 ×1、実データ4組、batch 1、128×128 cropで2回のパラメータ更新に成功 |
| 学習成果物 | タスク `SUCCEEDED`。約246 MBの重み・ログ・設定・summaryをGCSへ保存し、ダウンロード・整合性・重み更新を確認 |

**学習は実行経路の検証までです。30,000 stepの本学習は未実施で、全データ転送、GPU/batch設定、定期保存・再開処理の適合が残っています。** 推論には2 stepの試験重みではなく、既存の正式モデルを使用しています。

初回200.293秒は起動・再試行・通信・処理・ダウンロードを含む単回測定で、起動時間の保証値ではありません。

## 3. 起動・停止方法

社内サーバー **gpu01** の端末で操作します。既存コンテナを再利用できます。

```bash
docker start denoise-vertex-ui       # 社内画面を起動
docker logs --tail 100 denoise-vertex-ui
docker stop denoise-vertex-ui        # 社内画面を停止
```

Vertex が副本0で休止中なら、画面からの予測要求で自動起動し、画面側が再試行します。**画面の停止とクラウドの撤去は別操作です。** 確実に推論サービスを停止する場合は、Cloud Console の Endpoint で **Undeploy** します。その後の利用には再デプロイが必要です。

Scale To Zero は Preview です。30日間以上、無通信で副本0の状態が続くと自動 undeploy されます。再作成・復旧の詳細コマンドは詳細版に記載しています。

## 4. 費用は発生したか

**実 GPU を使った学習・推論、イメージ保存、GCS保存・通信を実行しており、課金対象の使用量は発生しています。実際の請求額は未確認です。**

GPU が副本0の間はその推論ノードの料金は発生しませんが、イメージ・学習成果物の保存費用等は残ります。課金時間には起動・待機等も含まれるため、画像処理46秒だけで費用を計算できません。学習タスクは終了済みです。

金額は [Cloud Billing](https://console.cloud.google.com/billing?project=hallowed-pipe-497802-i0) の Reports で、本プロジェクト・検証日・サービス/SKUを絞って確認します。反映遅延や割引・クレジットも確認が必要です。今回 Billing API は未有効の403でしたが、これは請求額が0、またはWeb画面も閲覧不能という意味ではありません。

## 5. 保存場所と管理画面

**AI経由でなくても、権限のある担当者が Google Cloud Console と Cloud Shell から管理できます。** `gcloud` はコマンドライン、Console はブラウザーの管理画面です。

| 対象 | 保存場所・確認先 |
|---|---|
| 推論 Endpoint | [Online prediction / Endpoints](https://console.cloud.google.com/vertex-ai/endpoints?project=hallowed-pipe-497802-i0)、東京、ID `8911645097217490944` |
| 学習タスク | [Training / Custom jobs](https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=hallowed-pipe-497802-i0)、ID `1044436952901746688` |
| Docker イメージ | [Artifact Registry / denoise](https://console.cloud.google.com/artifacts/docker/hallowed-pipe-497802-i0/asia-northeast1/denoise?project=hallowed-pipe-497802-i0)、`denoise-vertex` と `denoise-vertex-inference`、タグ `20261117` |
| 学習成果物 | [Cloud Storage](https://console.cloud.google.com/storage/browser/hallowed-pipe-497802-i0-denoise-vertex?project=hallowed-pipe-497802-i0)、`vertex-smoke/20261117-run1/` |
| ソースコード | gpu01 の `/mnt/home/huang/code/cst_ai/BasicSR`。本デプロイの変更は **Gitに未コミット・未push** |

**イメージのアップロード先とソースコードのGitリポジトリは別です。** 実行用コードはイメージ内に含まれますが、イメージをpushしてもGitは更新されません。コードの正式な共有・引き継ぎは未完了です。

現在の Gradio は推論専用で、学習開始ボタンはありません。学習は Google の管理画面／Cloud Shell で操作でき、検証済みの2 step学習の再実行コマンドも詳細版にあります。閲覧・操作には担当者自身のIAM権限が必要です。

## 6. 現在の確認事項

過去の推論・学習・自動起動・縮小テストは成功しています。ただし、**21:10 JSTの読み取り確認では、Endpoint に汎用的な `System error` メッセージがありました。** 副本数は直近指標で0、社内画面はHTTP 200ですが、この確認では再度予測を実行していません。**デモ前の再接続・推論確認が必要です。** エラー原因は未確定です。

詳細な起動・復旧・学習再実行コマンドと証跡は、[詳細版](vertex-operations-guide.zh-CN.md)を参照してください。本要約の作成に伴う新規GPUテストは実施していません。
