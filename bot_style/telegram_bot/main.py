from model import StyleTransferModel
from telegram_token import token
import numpy as np
from PIL import Image
from io import BytesIO
import telegram
from fast_style import FastStylizer

# В бейзлайне пример того, как мы можем обрабатывать две картинки, пришедшие от пользователя.
# При реалиазации первого алгоритма это Вам не понадобится, так что можете убрать загрузку второй картинки.
# Если решите делать модель, переносящую любой стиль, то просто вернете код)

model = StyleTransferModel()
fast_model = FastStylizer()
first_image_file = {}
model_list = {}
FIRST, SECOND, THIRD, OWN_STYLE, FAST, PHOTO_WAIT = range(6)


def send_prediction_on_photo(bot, update):
    # Нам нужно получить две картинки, чтобы произвести перенос стиля, но каждая картинка приходит в
    # отдельном апдейте, поэтому в простейшем случае мы будем сохранять id первой картинки в память,
    # чтобы, когда уже придет вторая, мы могли загрузить в память уже сами картинки и обработать их.
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)

    if chat_id in first_image_file:

        # первая картинка, которая к нам пришла станет content image, а вторая style image
        content_image_stream = BytesIO()

        first_image_file[chat_id].download(out=content_image_stream)
        del first_image_file[chat_id]
        style_image_stream = BytesIO()
        image_file.download(out=style_image_stream)
        output = model.transfer_style(content_image_stream, style_image_stream)

        # теперь отправим назад фото
        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")
    else:
        first_image_file[chat_id] = image_file


def send_prediction_on_photo_fast(bot, update):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)

    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)
    output = fast_model.stylize(model_list[chat_id], content_image_stream)

    # теперь отправим назад фото
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to user")


def start(bot, update):
    update.message.reply_text(main_menu_message(),
                              reply_markup=main_menu_keyboard())
    return THIRD


def main_menu(bot, update):
    query = update.callback_query
    bot.edit_message_text(chat_id=query.message.chat_id,
                          message_id=query.message.message_id,
                          text=main_menu_message(),
                          reply_markup=main_menu_keyboard())


def main_menu_message():
    return 'Choose the option in main menu:'


def first(bot, update):
    print("FIRST")
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text=u"OK, upload content and style images, and let the magic happen!(Be patient)"
    )
    return OWN_STYLE


def second(bot, update):
    print("SECOND")
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text="Now, pick the style",
        reply_markup=models_menu_keyboard())
    return FAST


def set_model(bot, update):
    print("SMODEL")
    query = update.callback_query
    print(query)
    chat_id = query.message.chat_id
    model_name = query.data
    model_list.update([(chat_id, model_name)])
    return PHOTO_WAIT


def models_menu_keyboard():
    keyboard = [[telegram.InlineKeyboardButton('Candy', callback_data="Candy")],
                [telegram.InlineKeyboardButton('Starry night', callback_data="Starry")]]
    return telegram.InlineKeyboardMarkup(keyboard)


def main_menu_keyboard():
    keyboard = [[telegram.InlineKeyboardButton('Use own style', callback_data=str(FIRST))],
                [telegram.InlineKeyboardButton('Use pre-trained models', callback_data=str(SECOND))]]
    return telegram.InlineKeyboardMarkup(keyboard)


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, CallbackQueryHandler, ConversationHandler
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    # используем прокси, так как без него у меня ничего не работало.
    # если есть проблемы с подключением, то попробуйте убрать прокси или сменить на другой
    # проекси ищется в гугле как "socks4 proxy"
    updater = Updater(token=token)

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            FIRST: [CallbackQueryHandler(first)],
            SECOND: [CallbackQueryHandler(second)],
            THIRD: [CallbackQueryHandler(first, pattern=str(FIRST)),
                    CallbackQueryHandler(second)],
            OWN_STYLE: [MessageHandler(Filters.photo, send_prediction_on_photo)],
            FAST: [CallbackQueryHandler(set_model, pattern='^.+$')],
            PHOTO_WAIT: [MessageHandler(Filters.photo, send_prediction_on_photo_fast)]
        },
        fallbacks=[CommandHandler('start', start)]
    )

    updater.dispatcher.add_handler(conv_handler)

    updater.start_polling()
