import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from IOUtils import read_params

# Считывание параметров из json-файла
params = read_params('params.json')

# Нужно ли использовать GPU или CPU
device = 'cuda:0' if torch.cuda.is_available() and params['gpu'] else 'cpu'

# Предобученный токенизатор
tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
# Предобученная модель
model = AutoModel.from_pretrained(params['model_name'])
model.to(device)

# Процедура токенизации и кодирования входного текста и превращения его в тензоры
def encode_text(texts):
    try:
        tokens = tokenizer(texts,
                           max_length=params['max_text_len'],   # Максимальная длина текста
                           truncation=True,                     # Если текст превысил максимальную длину, обрезать его
                           padding='max_length',                # Дополнять текст до максимальной длины
                           return_tensors='pt')                 # Возвращаем тензор pytorch
        tokens.to(device)
    except Exception:
        tokens = None
    return tokens

# Процедура осреднения векторов ембеддинга
# model_output - тензор векторов ембеддинга, получившийся на выходе последнего слоя модели
# attention_mask - тензор-маска исходного текста
def mean_pooling(model_output, attention_mask):
    embeddings = model_output.last_hidden_state
    attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    mask_embeddings = embeddings * attention_mask
    summed = torch.sum(mask_embeddings, 1)
    counts = torch.clamp(attention_mask.sum(1), min=1e-9)
    return summed / counts


"""
Определение схожести двух текстов на основе осреднения ембеддингов с подсчетом косинусного расстояния
На вход подаются два текста
На выходе - вещественное занчение от 0 до 1, показывающее насколько близки по смыслу тексты
0 - тексты очень далеки по смыслу
1 - тексты полностью идентичны по смыслу
"""
def find_similarity(text_1: str, text_2: str) -> float:
    # Токенизируем строку и кодируем токены
    tokens = encode_text([text_1, text_2])
    if not tokens:
        return 2

    try:
        with torch.no_grad():
            # Вычисляем вектора ембеддингов
            outputs = model(**tokens)
    except Exception:
        outputs = None
    if not outputs:
        return 3

    try:
        # Осредняем вектора ембеддингов по токенам текста
        mean = mean_pooling(outputs, tokens['attention_mask'])
        mean = mean.cpu().detach().numpy()

        # Вычисляем косинусное расстояние между двумя осредненными векторами
        similarity = cosine_similarity([mean[0]], [mean[1]])[0][0]
    except Exception:
        return 4
    return float(similarity)