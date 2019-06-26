FROM python:3.7
RUN pip3 install python-telegram-bot
RUN pip3 install numpy
RUN pip3 install Pillow
RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
RUN pip3 install torchvision
RUN mkdir /bot_style
ADD . /bot_style
WORKDIR /bot_style/bot_style/telegram_bot
CMD python main.py
