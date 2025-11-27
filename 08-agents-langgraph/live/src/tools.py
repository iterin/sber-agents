"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
from typing import Dict

import requests
from langchain_core.tools import tool

import rag
from config import config

logger = logging.getLogger(__name__)


@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).

    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)

        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)

        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content,  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)

        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)


SUPPORTED_CURRENCIES = {"USD", "EUR", "RUB"}


def _fetch_rates(base_currency: str) -> Dict[str, float]:
    """
    Получает курсы валют с exchangerate-api.com для выбранной базовой валюты.
    """
    if not config.EXCHANGERATE_API_KEY:
        raise RuntimeError("EXCHANGERATE_API_KEY is not configured")

    url = f"https://v6.exchangerate-api.com/v6/{config.EXCHANGERATE_API_KEY}/latest/{base_currency}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        # API вернуло ошибку
        error_type = data.get("error-type", "unknown_error")
        raise RuntimeError(f"ExchangeRate API error: {error_type}")

    return data.get("conversion_rates", {})


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует сумму из одной валюты в другую, используя exchangerate-api.com.

    Args:
        amount: Сумма для конвертации.
        from_currency: Исходная валюта (USD, EUR, RUB).
        to_currency: Целевая валюта (USD, EUR, RUB).

    Returns:
        JSON-строка с результатом конвертации:
        {
          "from_currency": "...",
          "to_currency": "...",
          "amount": <исходная сумма>,
          "rate": <курс>,
          "converted_amount": <конвертированная сумма>
        }
        В случае ошибки возвращается JSON с полем "error".
    """
    try:
        if amount < 0:
            return json.dumps(
                {"error": "Сумма для конвертации не может быть отрицательной."},
                ensure_ascii=False,
            )

        from_cur = from_currency.upper()
        to_cur = to_currency.upper()

        if from_cur not in SUPPORTED_CURRENCIES or to_cur not in SUPPORTED_CURRENCIES:
            return json.dumps(
                {
                    "error": (
                        "Поддерживаются только валюты USD, EUR, RUB. "
                        f"Получено from_currency={from_cur}, to_currency={to_cur}."
                    )
                },
                ensure_ascii=False,
            )

        if from_cur == to_cur:
            # Ничего конвертировать не нужно
            return json.dumps(
                {
                    "from_currency": from_cur,
                    "to_currency": to_cur,
                    "amount": amount,
                    "rate": 1.0,
                    "converted_amount": amount,
                },
                ensure_ascii=False,
            )

        # Получаем курсы для базовой валюты from_cur
        rates = _fetch_rates(from_cur)
        if to_cur not in rates:
            return json.dumps(
                {
                    "error": (
                        f"Не удалось получить курс {from_cur}->{to_cur} "
                        "с сервиса exchangerate-api.com."
                    )
                },
                ensure_ascii=False,
            )

        rate = float(rates[to_cur])
        converted_amount = round(amount * rate, 2)

        result = {
            "from_currency": from_cur,
            "to_currency": to_cur,
            "amount": amount,
            "rate": rate,
            "converted_amount": converted_amount,
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in currency_converter: {e}", exc_info=True)
        return json.dumps(
            {"error": "Ошибка при конвертации валют. Попробуйте позже."},
            ensure_ascii=False,
        )
