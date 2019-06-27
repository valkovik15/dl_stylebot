import re
from PIL import Image
import torch
from torchvision import transforms
import torch.onnx
from utils import *
from transformer import TransformerNet
from deploy_config import *
class FastStylizer():
    '''Класс, производящий перенос стиля'''

    def __init__(self):
        '''Инициализация всех доступных моделей в словарь'''
        self.zoo = {}
        for key in model_paths:
            self.zoo.update([(key, torch.load(model_paths[key]))])

    def stylize(self, key, image):
        '''Перенос стиля с помощью предобученного TransformNeta'''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        content_image = Image.open(image)
        loader = transforms.Compose([
            transforms.ToTensor()])  # превращаем в удобный формат
        content_image = loader(content_image)
        content_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_size),
            transforms.ToTensor(),

            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = self.zoo[key]
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            output = style_model(content_image).cpu()
        return get_image(output[0])
