import asyncio
import base64
import json
import logging
import tempfile
from datetime import time
from pathlib import Path
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
import ffmpeg
import soundfile as sf
from vosk import KaldiRecognizer, Model as VoskModel
from openai import APIError, InternalServerError, NotFoundError
from llm import get_transaction_response_text, get_transaction_response_image
from models import Transaction
from config import config

logger = logging.getLogger(__name__)
router = Router()

# Глобальные словари для хранения данных
chat_conversations: dict[int, list[dict]] = {}
transactions: dict[int, list[Transaction]] = {}

# Максимальная длина сообщения пользователя
MAX_MESSAGE_LENGTH = 4000
_vosk_model: VoskModel | None = None
_vosk_model_lock = asyncio.Lock()


async def ensure_vosk_model() -> VoskModel:
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model

    async with _vosk_model_lock:
        if _vosk_model is not None:
            return _vosk_model

        model_path = Path(config.VOSK_MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Vosk model not found at '{model_path}'. "
                "Download a Russian model, e.g.: "
                "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip "
                "and unpack to models/vosk-ru-small"
            )
        logger.info("Loading Vosk model from '%s'...", model_path)
        _vosk_model = await asyncio.to_thread(lambda: VoskModel(str(model_path)))
    return _vosk_model


def _convert_ogg_bytes_to_wav(ogg_bytes: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_file:
        ogg_file.write(ogg_bytes)
        ogg_path = Path(ogg_file.name)

    wav_path = ogg_path.with_suffix(".wav")
    try:
        (
            ffmpeg
            .input(str(ogg_path))
            .output(
                str(wav_path),
                format="wav",
                acodec="pcm_s16le",
                ac=1,
                ar="16000",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    finally:
        ogg_path.unlink(missing_ok=True)

    return wav_path


async def convert_ogg_bytes_to_wav(ogg_bytes: bytes) -> Path:
    return await asyncio.to_thread(_convert_ogg_bytes_to_wav, ogg_bytes)


def _transcribe_with_vosk(model: VoskModel, wav_path: Path) -> str:
    audio_data, samplerate = sf.read(str(wav_path), dtype="int16")
    if samplerate != 16000:
        raise RuntimeError(f"Unexpected samplerate {samplerate}, expected 16000")
    recognizer = KaldiRecognizer(model, samplerate)
    recognizer.SetWords(False)
    if recognizer.AcceptWaveform(audio_data.tobytes()):
        result = json.loads(recognizer.Result())
    else:
        result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()


async def transcribe_audio(wav_path: Path) -> str:
    model = await ensure_vosk_model()
    return await asyncio.to_thread(_transcribe_with_vosk, model, wav_path)

@router.message(Command("start"))
async def cmd_start(message: Message):
    chat_id = message.chat.id
    logger.info(f"User {chat_id} started the bot")
    
    # Очищаем историю и транзакции для данного чата
    chat_conversations[chat_id] = [
        {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
    ]
    transactions[chat_id] = []
    
    await message.answer(
        "Привет! Я персональный финансовый советник.\n\n"
        "Я могу:\n"
        "• Извлекать транзакции из ваших сообщений\n"
        "• Вести учет доходов и расходов\n"
        "• Предоставлять советы по управлению финансами\n\n"
        "Используйте /start для начала нового диалога и очистки истории."
    )

@router.message(Command("balance"))
async def cmd_balance(message: Message):
    chat_id = message.chat.id
    logger.info(f"Balance requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "💵 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Расчет баланса, доходов и расходов
    total_income = sum(t.amount for t in user_transactions if t.type.value == "income")
    total_expense = sum(t.amount for t in user_transactions if t.type.value == "expense")
    balance = total_income - total_expense
    
    # Статистика по категориям
    category_stats: dict[str, float] = {}
    for t in user_transactions:
        category = t.category
        if category not in category_stats:
            category_stats[category] = 0.0
        if t.type.value == "income":
            category_stats[category] += t.amount
        else:
            category_stats[category] -= t.amount
    
    # Форматирование отчета
    report_lines = [
        "💵 **Отчет о балансе**\n",
        f"📊 Баланс: {balance:.2f} руб.",
        f"💰 Доходы: {total_income:.2f} руб.",
        f"💸 Расходы: {total_expense:.2f} руб.",
        f"\n📈 Всего транзакций: {len(user_transactions)}",
        "\n**Статистика по категориям:**"
    ]
    
    # Сортируем категории по сумме (от большей к меньшей)
    sorted_categories = sorted(category_stats.items(), key=lambda x: abs(x[1]), reverse=True)
    for category, amount in sorted_categories:
        sign = "💰" if amount > 0 else "💸"
        report_lines.append(f"{sign} {category}: {amount:+.2f} руб.")
    
    await message.answer("\n".join(report_lines))

@router.message(Command("transactions"))
async def cmd_transactions(message: Message):
    chat_id = message.chat.id
    logger.info(f"Transactions list requested by {chat_id}")
    
    # Получаем транзакции пользователя
    user_transactions = transactions.get(chat_id, [])
    
    if not user_transactions:
        await message.answer(
            "📋 У вас пока нет транзакций.\n\n"
            "Отправьте сообщение с транзакцией или изображение чека для начала учета."
        )
        return
    
    # Сортируем транзакции по дате (от новых к старым)
    sorted_transactions = sorted(user_transactions, key=lambda t: (t.date, t.time or time(0, 0)), reverse=True)
    
    # Форматирование списка транзакций
    report_lines = [
        f"📋 **Все транзакции** ({len(user_transactions)} шт.)\n"
    ]
    
    for i, t in enumerate(sorted_transactions, 1):
        # Форматирование даты и времени
        date_str = t.date.strftime("%d.%m.%Y")
        time_str = f" {t.time.strftime('%H:%M')}" if t.time else ""
        
        # Знак и тип транзакции
        sign = "💰" if t.type.value == "income" else "💸"
        type_str = "Доход" if t.type.value == "income" else "Расход"
        
        # Форматирование суммы
        amount_str = f"{t.amount:.2f}".rstrip('0').rstrip('.')
        
        # Описание (если есть)
        desc_str = f"\n   {t.description}" if t.description else ""
        
        report_lines.append(
            f"{i}. {sign} **{type_str}** {amount_str} руб.\n"
            f"   📅 {date_str}{time_str}\n"
            f"   🏷️ {t.category}{desc_str}"
        )
    
    # Если транзакций много, разбиваем на несколько сообщений (Telegram лимит ~4096 символов)
    report_text = "\n\n".join(report_lines)
    if len(report_text) > 4000:
        # Разбиваем на части
        parts = []
        current_part = [report_lines[0]]  # Заголовок
        current_length = len(report_lines[0])
        
        for line in report_lines[1:]:
            line_length = len(line) + 2  # +2 для "\n\n"
            if current_length + line_length > 4000:
                parts.append("\n\n".join(current_part))
                current_part = [line]
                current_length = len(line)
            else:
                current_part.append(line)
                current_length += line_length
        
        if current_part:
            parts.append("\n\n".join(current_part))
        
        # Отправляем части
        for part in parts:
            await message.answer(part)
    else:
        await message.answer(report_text)

@router.message(lambda message: message.photo or (message.document and message.document.mime_type and message.document.mime_type.startswith("image/")))
async def handle_image(message: Message):
    chat_id = message.chat.id
    
    logger.info(f"Image received from {chat_id}")
    
    # Инициализируем историю если её нет
    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_IMAGE}
        ]
    
    try:
        # Определяем источник изображения
        if message.photo:
            # Берем самое большое изображение
            photo = message.photo[-1]
            file_info = await message.bot.get_file(photo.file_id)
        elif message.document:
            file_info = await message.bot.get_file(message.document.file_id)
        else:
            await message.answer("Не удалось обработать изображение.")
            return
        
        # Скачиваем изображение
        file_buffer = await message.bot.download_file(file_info.file_path)
        image_bytes = file_buffer.getvalue()
        
        # Конвертируем в base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Получаем историю сообщений без системного промпта для контекста
        message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []
        
        # Получаем ответ LLM с structured output
        response = await get_transaction_response_image(image_base64, message_history)
        
        # Детальное логирование ответа LLM
        logger.info(f"LLM response for image from {chat_id}: answer='{response.answer[:200]}...', transactions_count={len(response.transactions)}")
        if response.transactions:
            logger.info(f"Extracted {len(response.transactions)} transactions from image for {chat_id}: {[t.model_dump() for t in response.transactions]}")
        else:
            logger.warning(f"No transactions extracted from image for {chat_id}")
        
        # Сохраняем транзакции
        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)
        
        # Рассчитываем баланс
        balance = sum(
            t.amount if t.type.value == "income" else -t.amount 
            for t in transactions.get(chat_id, [])
        )
        
        # Формируем ответ пользователю
        answer_text = response.answer
        
        # Добавляем статус транзакций
        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"
        
        # Добавляем баланс
        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."
        
        # Добавляем изображение в историю как текстовое описание (для контекста)
        chat_conversations[chat_id].append(
            {"role": "user", "content": "[Изображение: чек/скриншот]"}
        )
        
        # Добавляем ответ LLM в историю
        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )
        
        await message.answer(answer_text)
    except (APIError, InternalServerError, NotFoundError) as e:
        logger.error(f"LLM API error for image from {chat_id}: {e}", exc_info=True)
        error_message = str(e)
        if "image input" in error_message.lower() or "404" in error_message or "not found" in error_message.lower():
            await message.answer(
                "Извините, используемая модель не поддерживает обработку изображений.\n\n"
                "Для работы с изображениями необходимо использовать vision-модель, например:\n"
                "• meta-llama/llama-3.2-11b-vision-instruct (OpenRouter)\n"
                "• llama3.2-vision (Ollama)\n\n"
                "Измените MODEL в файле .env на одну из этих моделей."
            )
        else:
            await message.answer(
                "Извините, произошла ошибка на стороне провайдера LLM при обработке изображения. "
                "Пожалуйста, попробуйте еще раз через несколько секунд."
            )
    except Exception as e:
        logger.error(f"Error processing image from {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке изображения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )

@router.message(lambda message: message.voice is not None)
async def handle_voice(message: Message):
    chat_id = message.chat.id
    logger.info(f"Voice message received from {chat_id}")

    try:
        file_info = await message.bot.get_file(message.voice.file_id)
        file_buffer = await message.bot.download_file(file_info.file_path)
        wav_path = await convert_ogg_bytes_to_wav(file_buffer.getvalue())
    except ffmpeg.Error as e:
        logger.error(f"Failed to convert voice message for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Не удалось обработать голосовое сообщение. Убедитесь, что формат поддерживается, и попробуйте снова."
        )
        return
    except Exception as e:
        logger.error(f"Error downloading voice message from {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Не удалось скачать голосовое сообщение. Попробуйте еще раз позже."
        )
        return

    try:
        transcription = await transcribe_audio(wav_path)
    except Exception as e:
        logger.error(f"Voice transcription failed for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Не удалось распознать голосовое сообщение. Попробуйте записать его еще раз."
        )
        return
    finally:
        wav_path.unlink(missing_ok=True)

    if not transcription:
        await message.answer(
            "Не удалось распознать текст в голосовом сообщении. Попробуйте записать его еще раз."
        )
        return

    await message.answer(f"🗣️ Расшифровал ваш голос:\n{transcription}")

    if len(transcription) > MAX_MESSAGE_LENGTH:
        await message.answer(
            "Распознанный текст слишком длинный. Пожалуйста, отправьте более короткое сообщение."
        )
        return

    await process_text_message(message, transcription)


async def process_text_message(message: Message, user_text: str) -> None:
    chat_id = message.chat.id
    logger.info(f"Message from {chat_id}: {user_text[:100]}...")

    if chat_id not in chat_conversations:
        chat_conversations[chat_id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT_TEXT}
        ]

    message_history = chat_conversations[chat_id][1:] if chat_conversations[chat_id] else []

    try:
        response = await get_transaction_response_text(user_text, message_history)

        logger.info(
            f"LLM response for {chat_id}: answer='{response.answer[:200]}...', "
            f"transactions_count={len(response.transactions)}"
        )
        if response.transactions:
            logger.info(
                f"Extracted {len(response.transactions)} transactions for {chat_id}: "
                f"{[t.model_dump() for t in response.transactions]}"
            )
        else:
            logger.warning(f"No transactions extracted from message: '{user_text}' for {chat_id}")

        if response.transactions:
            if chat_id not in transactions:
                transactions[chat_id] = []
            transactions[chat_id].extend(response.transactions)

        balance = sum(
            t.amount if t.type.value == "income" else -t.amount
            for t in transactions.get(chat_id, [])
        )

        answer_text = response.answer

        if response.transactions:
            count = len(response.transactions)
            answer_text += f"\n\n✅ Найдено и сохранено {count} транзакция{'и' if count > 1 else ''}"
        else:
            answer_text += "\n\nℹ️ Транзакции не найдены"

        balance_str = f"{balance:.0f}" if balance == int(balance) else f"{balance:.2f}"
        answer_text += f"\n💵 Баланс: {balance_str} руб."

        chat_conversations[chat_id].append(
            {"role": "user", "content": user_text}
        )

        chat_conversations[chat_id].append(
            {"role": "assistant", "content": response.answer}
        )

        await message.answer(answer_text)
    except (APIError, InternalServerError) as e:
        logger.error(f"LLM API error for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Извините, произошла ошибка на стороне провайдера LLM. "
            "Пожалуйста, попробуйте еще раз через несколько секунд."
        )
    except Exception as e:
        logger.error(f"Error in process_text_message for {chat_id}: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )


@router.message()
async def handle_message(message: Message):
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return

    if len(message.text) > MAX_MESSAGE_LENGTH:
        await message.answer(
            f"Извините, ваше сообщение слишком длинное ({len(message.text)} символов). "
            f"Максимальная длина: {MAX_MESSAGE_LENGTH} символов."
        )
        return

    await process_text_message(message, message.text)