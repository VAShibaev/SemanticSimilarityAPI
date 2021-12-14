from fastapi import FastAPI

from SearchSimilarity import find_similarity

app = FastAPI()

@app.post('/similar-recognition')
def compute_similarity(text_1: str, text_2: str):
    # Вычисление сходства между двумя текстами
    similarity = find_similarity(text_1, text_2)
    if similarity <= 1:
        return {'similarity': similarity}
    # Если схожесть текстов чуть превысила 1, округляем ее до 1
    elif similarity < 1.1:
        return {'similarity': 1.0}
    # Если в ходе вычисления схожести возникла ошибка, отправляем клиенту ответ с уведомлением об ошибке
    else:
        return {
            'Error': 1,
            'description': 'Error during computing similarity'
        }
        

# uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# docker build --no-cache -t sim-api .
# docker images
# docker run -t -p 8002:8080 sim-api

# docker ps
# docker builder prune -f