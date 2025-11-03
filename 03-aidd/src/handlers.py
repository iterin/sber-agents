from aiogram import Router, types
from aiogram.filters import Command
from src.llm import generate_reply

router = Router()


@router.message(Command("ping"))
async def handle_ping(message: types.Message) -> None:
    await message.answer("pong")


@router.message()
async def handle_text(message: types.Message) -> None:
    if not message.text:
        return
    reply = generate_reply(message.text)
    if reply:
        await message.answer(reply)


