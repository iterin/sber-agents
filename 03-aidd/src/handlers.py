from aiogram import Router, types
from aiogram.filters import Command
from src.llm import generate_reply
from src.memory import get_history, reset_history

router = Router()


@router.message(Command("ping"))
async def handle_ping(message: types.Message) -> None:
    await message.answer("pong")


@router.message(Command("reset"))
async def handle_reset(message: types.Message) -> None:
    reset_history(message.chat.id)
    await message.answer("Контекст очищен.")


@router.message()
async def handle_text(message: types.Message) -> None:
    if not message.text:
        return
    history = get_history(message.chat.id)
    reply = generate_reply(message.text, history)
    # обновляем историю
    history.append(("user", message.text))
    history.append(("assistant", reply))
    if reply:
        await message.answer(reply)


