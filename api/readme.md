# Python FastAPI Project with LangChain & Google ‎Gemini AI for RAG

Windows 11 & Python 3.11.0  

## Key Features

- Implemented routing
- Implemented multi queries
- Implemented adaptive retrieval
- Implemented reranking
- Trying to make raptor ..
  
## Usage
0. Clone the repository to your local machine.

1. Obtain your Gemini and Pinecone API keys. Create a .env file with the following content: 

```
GOOGLE_API_KEY =  your_api
PINECONE_API_KEY = your_api
```

2. Create pythonn env: 
```
python -m venv venv
```
3. Install Requirements:
```
venv\Scripts\activate
python -m pip install -r requirements.txt
```
4. Run the FastAPI server:
```
python main.py
```
5. View the Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) or trying chat: [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

