"""
Предварительное скачивание библиотекой transformers
всех необходимых моделей при сборке docker-образа
"""

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
