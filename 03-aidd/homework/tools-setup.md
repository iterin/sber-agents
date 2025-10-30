# Установка инструментов

## 📋 Список необходимых инструментов

- ✅ **Cursor AI** (или другая IDE с AI-ассистентом: Windsurf, VSCode + CLI агенты)
- ✅ **uv** - пакетный менеджер Python
- ✅ **Python 3.11+**
- ✅ **Git**
- ✅ **make**

### Опционально:
- **Warp Terminal** - для упрощения установки

---

## 🚀 Быстрая установка

### 1. Cursor (или другая IDE)

**Cursor (рекомендуется):**
1. Скачайте с [cursor.com](https://www.cursor.com/)
2. Установите и войдите в аккаунт
3. **Активируйте платную подписку** (опционально, но рекомендуется для доступа к мощным моделям)

**Альтернативы:**
- **Windsurf** - [windsurf.com](https://codeium.com/windsurf)
- **VSCode** + CLI агенты (Claude, GigaCode, Codex)
- Любая другая IDE с поддержкой AI-ассистентов

> **Примечание:** В этом задании промпты и примеры написаны для работы в Cursor, но вы можете адаптировать их для любого инструмента с AI-ассистентом.

---

### 2. Основные инструменты

#### Через Warp Terminal (рекомендуется)

Установите [Warp](https://www.warp.dev/), затем используйте Agent Mode (✨):

```bash
✨ установи uv пакетный менеджер python
✨ установи python 3.11
```

#### Ручная установка

**uv:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Python через uv:**
```bash
uv python install 3.11
```

**Git:**
```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt-get install git

# Windows - скачайте с git-scm.com
```

**make:**
```bash
# macOS
brew install make

# Ubuntu/Debian
sudo apt-get install build-essential

# Windows
choco install make
# или используйте Git Bash (make уже включен)
```

---

## ✅ Проверка установки

Выполните команды для проверки:

```bash
# Основные инструменты
uv --version          # должно быть 0.4+
python --version      # должно быть 3.11+
git --version
make --version
```

**Cursor (или ваша IDE):**
- Откройте вашу IDE
- Проверьте доступ к AI-ассистенту
- (Для Cursor) Проверьте доступ к моделям (Cmd/Ctrl+L → выбор модели)

> **Примечание о моделях:** Используйте модель в зависимости от вашей подписки. Для эффективной AI-driven разработки рекомендуются:
> - Claude 3.5 Sonnet (платная подписка Cursor)
> - GPT-4 (при наличии доступа)
> - Бесплатные альтернативы: Claude 3.5 Haiku, GPT-3.5

---

## 🔧 Конфигурация Git

```bash
git config --global user.name "Ваше Имя"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

---

