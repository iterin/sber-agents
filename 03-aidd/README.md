# 03-aidd — LLM-бот (MVP)

Минимальный Telegram-бот на aiogram с командами и LLM-интеграцией (дальше по плану).

## Быстрый старт
1. Установите Python 3.10+
2. Создайте `.env` на основе `.env.example` (минимум: `TELEGRAM_BOT_TOKEN=...`; при необходимости: `TELEGRAM_PROXY_URL=http://proxy.server:3128` или `socks5://host:port`; чтобы использовать системные прокси/сертификаты, добавьте `TELEGRAM_TRUST_ENV=1`)
3. Установите зависимости и запустите:

```bash
make run
```

При проблемах с подключением:
- Убедитесь, что установлен `aiohttp-socks` (ставится автоматически из зависимостей)
- Укажите `TELEGRAM_PROXY_URL` (HTTP/HTTPS или socks5) и перезапустите

## LLM (Итерация 2)
- Реальный ответ модели:
  - OpenAI: `OPENAI_API_KEY`, опционально `OPENAI_MODEL`
  - OpenRouter: можно использовать `OPENROUTER_API_KEY` (авто-настройка `OPENAI_BASE_URL=https://openrouter.ai/api/v1`), опционально `OPENROUTER_MODEL` или `OPENAI_MODEL`
- Без ключа бот отвечает в mock-режиме: `[mock] <ваш текст>`.
- При необходимости вручную задайте `OPENAI_BASE_URL` (например, `https://openrouter.ai/api/v1`).
- Для диагностики включите `DEBUG_LLM=1` (в консоль выводятся краткие статусы и ошибки).

## Структура
См. `docs/tasklist.md` и `docs/workflow.md`.

