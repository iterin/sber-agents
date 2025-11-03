import asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_PROXY_URL, validate_config
from src.handlers import router as handlers_router


async def main() -> None:
    validate_config()
    session = AiohttpSession(proxy=TELEGRAM_PROXY_URL or None)
    bot = Bot(token=TELEGRAM_BOT_TOKEN, session=session)
    dp = Dispatcher()
    dp.include_router(handlers_router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())


