import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from api.endpoints import face_beautify_endpoint



HOST=os.environ.get('HOST', '0.0.0.0')
PORT=int(os.environ.get('PORT', 8648))
SERVER_ENV=os.environ.get('SERVER_ENV', 'debug')


with open('app_description.html', 'r', encoding='utf-8') as file:
    description = file.read()

app = FastAPI(
    title='Face Beautify',
    description=description,
    version='1.0.0',
    # max_concurrency=5,
)

# allow all origins
origins = ["*"]    
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(face_beautify_endpoint.router, prefix="/api", tags=["face_beautify"])

@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    return HTMLResponse(f"<h1> {exc} </h1>", status_code=500)

@app.get("/")
def read_root():
    return HTMLResponse(description)


@app.get('/status', summary='status of server')
def status():
    return {
        'host': HOST,
        'port': PORT,
        'SERVER_ENV': SERVER_ENV,
        'Version': '1.1.0',
    }



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'app:app', # 'module_name:variable_name
        host=HOST,
        port=PORT,
        limit_concurrency=10,
        reload=True,
    )