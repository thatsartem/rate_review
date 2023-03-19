from fastapi import FastAPI, Form, Request
from typing import Optional
from services import get_review_out
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.post('/')
def rate(request: Request,review: Optional[str] = Form(...)):
    if review:
        text = review
        rev_out = get_review_out(text)
        message = f"rate: {rev_out.rate}, sentiment: {rev_out.sentiment}"
    else:
        message = "ENTER REVIEW"
    return templates.TemplateResponse("index.html", {"request": request, "message": message})

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    