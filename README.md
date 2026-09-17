# Local Decision Playground

ローカルLLMを分類器・意思決定器として試す、依存パッケージ不要の小さなWebアプリです。定義した選択肢だけを生成可能にし、各候補の `logprobs` を候補内で正規化して相対確率を表示します。テキスト、JSON、PNG/JPEG/WebP画像を入力できます。

[TypeSafe AIのJev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)が提供するdecision interfaceに着想を得ていますが、TypeSafe公式製品やJevのローカル実装ではありません。

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `allowed_token_ids`、`logprobs`、`/tokenize` に対応するvLLM系のOpenAI互換API
- 1つ以上のモデルを配信するAPI

## Run

```bash
cp .env.example .env
uv run --env-file .env server.py
```

ブラウザで <http://127.0.0.1:8768> を開きます。

`.env`の`MODEL_ENDPOINTS`には、モデルAPIが公開するモデル名とURLのJSON objectを指定します。キーはvLLMの`--served-model-name`、または`/v1/models`が返す`id`と一致させてください。`.env.example`は`qwen-local`を`http://127.0.0.1:8000`で配信する例です。

複数の質問は最大8本を同時にモデルAPIへ送ります。逐次実行に戻す場合は `QUESTION_WORKERS=1` を指定します。

このアプリが表示する値は、許可した候補間で再正規化した未校正の相対確率です。正答率や統計的に校正された信頼度ではありません。

`choice` と `score` のconfidence計算は、MIT Licenseの[TypeSafe System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python)を参考にしています。

## Files

- `index.html`: UI（HTML/CSS/JavaScriptのみ）
- `server.py`: 静的配信、モデルAPI中継、候補確率の計算
- `.env.example`: モデル接続とサーバー設定の例
