# dl_stylebot
DLSchool graduation project located at t.me/valkovikstylebot.

Возникшие проблемы:

1. От переноса своего стиля пришлось отказаться из-за малых мощностей Heroku(код оставлен), Google Cloud, кажется, не даёт белорусам не-бизнес акккаунтов, а других альтернатив я не нашёл

2. Некоторые модели слишком изменяли фотографию, ранние остановки обучения и изменения размеров батча ни к чему не приводили(например, рисунки Густава Климта, "Волна" Хоккусая, "Постоянство памяти" Дали). Наверно, стиль этих рисунков нельзя переносить, не учитывая специфику конкретного изображения

3. Бесплатный план Heroku переводит проект в режим сна, если к нему нет обращений в течение получаса. Загружается порядка минуты, медленнее, чем я ожидал
