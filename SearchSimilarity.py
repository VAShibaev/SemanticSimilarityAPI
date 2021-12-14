import torch
from bert_score import BERTScorer

from IOUtils import read_params

# Считывание параметров из json-файла
params = read_params('params.json')

# Нужно ли использовать GPU или CPU
device = 'cuda:0' if torch.cuda.is_available() and params['gpu'] else 'cpu'

# Создаем Scorer для вычисления схожести
scorer = BERTScorer(model_type=params['model'],     # Какую модель взять за основу
                    num_layers=params['layers'],    # После какого слоя модели брать ембеддинги
                    lang=params['language'],        # Для какого языка брать модель
                    device=device)                  # На каком устройстве считать модель

"""
Определение схожести двух текстов
На вход подаются два текста
На выходе - вещественное занчение от 0 до 1, показывающее насколько близки по смыслу тексты
0 - тексты очень далеки по смыслу
1 - тексты полностью идентичны по смыслу
"""
def find_similarity(text_1: str, text_2: str) -> float:
    try:
        # Вычисляем схожесть текстов
        P, _, _ = scorer.score([text_1], [text_2])
    except Exception:
        return 2.0
    return float(P.cpu().detach().numpy()[0])
