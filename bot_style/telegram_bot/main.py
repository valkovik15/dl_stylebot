# from model import StyleTransferModel
# from telegram_token import token
from io import BytesIO
import telegram
from fast_style import FastStylizer
import os
token = os.getenv("TOKEN") #Получаем из переменных Heroku
# model = StyleTransferModel()
fast_model = FastStylizer()
#first_image_file = {}
model_list = {} #Соответствие юзер - модель
FIRST, SECOND, THIRD, OWN_STYLE, FAST, PHOTO_WAIT = range(6)
# HEROKU не потянул:(
'''
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
'''


def send_prediction_on_photo_fast(bot, update):
    '''Прогонка изображений через объект FastModel'''
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
    bot.send_message(chat_id,
                     text="Thank you, come again!. Use /start to change model. Or send another photo and I will apply current style to it")


def start(bot, update):
    '''Вывод главного меню'''
    update.message.reply_text(main_menu_message(),
                              reply_markup=main_menu_keyboard())
    return THIRD


def main_menu_message():
    return 'Choose the option in main menu:'


def first(bot, update):
    '''Обработчик первого пункта меню'''
    print("FIRST")
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text=u"OK, upload content and style images, and let the magic happen!(Be patient)"
    )
    return OWN_STYLE


def second(bot, update):
    '''Обработчик второго пункта меню'''
    print("SECOND")
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text="Now, pick the style",
        reply_markup=models_menu_keyboard())
    return FAST


def set_model(bot, update):
    '''Достаем из запроса выбор пользователем стиля'''
    print("SMODEL")
    query = update.callback_query
    print(query)
    chat_id = query.message.chat_id
    model_name = query.data
    model_list.update([(chat_id, model_name)])
    return PHOTO_WAIT


def models_menu_keyboard():
    '''Формирование меню для стиля'''
    keyboard = [[telegram.InlineKeyboardButton('Amadeo', callback_data="Amadeo")],
        [telegram.InlineKeyboardButton('Candy', callback_data="Candy")],
                [telegram.InlineKeyboardButton('Starry night by Van Gogh', callback_data="Starry")],
                [telegram.InlineKeyboardButton('Princess', callback_data="Princess")],
                [telegram.InlineKeyboardButton('Mosaic', callback_data="Mosaic")],
                [telegram.InlineKeyboardButton('Udnie', callback_data="Udnie")],
                [telegram.InlineKeyboardButton('Monet', callback_data="Monet")],
                [telegram.InlineKeyboardButton('Scream by Munk', callback_data="Munk")],
                [telegram.InlineKeyboardButton('Paul Sérusier', callback_data="Serusier")],
                [telegram.InlineKeyboardButton('Gogen', callback_data="Gogen")],
                [telegram.InlineKeyboardButton('Petrov-Vodkin', callback_data="Petrov-Vodkin")],
                [telegram.InlineKeyboardButton('On a wild North by Shishkin', callback_data="Winter")]]
    return telegram.InlineKeyboardMarkup(keyboard)


def main_menu_keyboard():
    '''Формирование главного меню'''
    keyboard = [[telegram.InlineKeyboardButton('Use own style', callback_data=str(FIRST))],
                [telegram.InlineKeyboardButton('Use pre-trained models', callback_data=str(SECOND))]]
    return telegram.InlineKeyboardMarkup(keyboard)


def help_callback(bot, update):
    '''Обработчик команды /help'''
    update.message.reply_text(
        "This bot can either transfer style of your own photo to the other picture, or execute a fast style tranfer with prepared photos. Use /start to begin")


def donate(bot, update):
    '''Будем вызывать, если запускается не на локальной машине(т.е. будем)'''
    query = update.callback_query
    bot.edit_message_text(
        chat_id=query.message.chat_id,
        message_id=query.message.message_id,
        text="Oh, I can do this, but my servers can't:( Try my pretrained models",
        reply_markup=models_menu_keyboard())
    return SECOND


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, CallbackQueryHandler, ConversationHandler
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    updater = Updater(token=token)
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            FIRST: [CallbackQueryHandler(first)],
            SECOND: [CallbackQueryHandler(second)],
            THIRD: [CallbackQueryHandler(donate, pattern=str(FIRST)),
                    CallbackQueryHandler(second)],
            OWN_STYLE: [CallbackQueryHandler(donate, pattern=str(FIRST))],
            # OWN_STYLE: [MessageHandler(Filters.photo, send_prediction_on_photo)], Нормальный обработчик для своего стиля
            FAST: [CallbackQueryHandler(set_model, pattern='^.+$')],
            PHOTO_WAIT: [MessageHandler(Filters.photo, send_prediction_on_photo_fast)]
        },
        fallbacks=[CommandHandler('start', start)]
    )

    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(CommandHandler("help", help_callback))
    PORT = int(os.environ.get("PORT", "8443")) ###Избегаем ошибки Heroku R10
    HEROKU_APP_NAME = os.environ.get("HEROKU_APP_NAME")
    updater.start_webhook(listen="0.0.0.0",
                          port=PORT,
                          url_path=token)
    updater.bot.set_webhook("https://{}.herokuapp.com/{}".format(HEROKU_APP_NAME, token))
