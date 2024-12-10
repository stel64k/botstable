import logging
from telegram import Bot

# Глобальные переменные
telegram_bot = None
telegram_chat_id = None

def configure_telegram(token, chat_id):
    """
    Настройка глобальных переменных Telegram бота и чата.
    """
    global telegram_bot, telegram_chat_id
    telegram_bot = Bot(token=token)
    telegram_chat_id = chat_id

def send_telegram_message(message):
    """
    Отправка сообщения в Telegram.
    """
    try:
        if not telegram_bot or not telegram_chat_id:
            raise ValueError("Telegram bot and chat ID must be configured first.")
        telegram_bot.send_message(chat_id=telegram_chat_id, text=message)
        logging.info(f"Telegram message sent: {message}")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")
