from fastapi import FastAPI
import requests
from fastapi import Response
import uvicorn 

app = FastAPI()

@app.get('/')
def home():
    return {"health": "healthy"} 

@app.get('/chat/v1/completions')
def chat_completion(
    prompt: str 
):
    res = requests.post(
        url = 'http://localhost:11434/api/generate',    
        json =  {
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False,   
        }
    )

    return Response(content=res.text, media_type='application/json')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)    