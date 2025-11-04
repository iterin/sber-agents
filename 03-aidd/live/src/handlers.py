import asyncio
import logging
from contextlib import suppress

from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from llm import get_response
from config import config

logger = logging.getLogger(__name__)
router = Router()

# Глобальный словарь для хранения историй диалогов
chat_conversations: dict[int, list[dict]] = {}

# Максимальная длина сообщения пользователя
MAX_MESSAGE_LENGTH = 4000


def _initialize_history(chat_id: int) -> None:
    chat_conversations[chat_id] = [
        {"role": "system", "content": config.SYSTEM_PROMPT}
    ]


async def _typing_indicator(bot, chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"User {message.chat.id} started the bot")

    # Инициализируем историю с системным промптом
    _initialize_history(message.chat.id)

    await message.answer(
        "Привет! Я LLM-ассистент.\n\n"
        "Я могу:\n"
        "• Отвечать на вопросы\n"
        "• Помогать с кодом\n"
        "• Писать тексты\n"
        "• Поддерживать диалог с учетом контекста\n\n"
        "Используйте /start для начала нового диалога."
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    logger.info(f"User {message.chat.id} requested help")
    await message.answer(
        "Доступные команды:\n"
        "• /start — начать новый диалог и получить приветствие\n"
        "• /help — показать эту подсказку\n"
        "• /reset — очистить текущий контекст диалога\n\n"
        "Просто отправьте сообщение, и я отвечу с учетом контекста беседы."
    )


@router.message(Command("reset"))
async def cmd_reset(message: Message):
    logger.info(f"User {message.chat.id} reset the conversation")
    _initialize_history(message.chat.id)
    await message.answer(
        "Контекст диалога очищен. Можно продолжать разговор."
    )


@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста (стикеры, фото и т.д.)
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return
    
    # Проверяем длину сообщения
    if len(message.text) > MAX_MESSAGE_LENGTH:
        await message.answer(
            f"Извините, ваше сообщение слишком длинное ({len(message.text)} символов). "
            f"Максимальная длина: {MAX_MESSAGE_LENGTH} символов."
        )
        return
    
    logger.info(f"Message from {message.chat.id}: {message.text[:100]}...")
    
    # Инициализируем историю если её нет
    if message.chat.id not in chat_conversations:
        _initialize_history(message.chat.id)
    
    # Добавляем сообщение пользователя в историю
    chat_conversations[message.chat.id].append(
        {"role": "user", "content": message.text}
    )
    
    try:
        # Получаем ответ LLM со всей историей
        typing_task = asyncio.create_task(
            _typing_indicator(message.bot, message.chat.id)
        )
        try:
            response = await get_response(chat_conversations[message.chat.id])
        finally:
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task

        # Добавляем ответ LLM в историю
        chat_conversations[message.chat.id].append(
            {"role": "assistant", "content": response}
        )
        
        await message.answer(response)
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )

