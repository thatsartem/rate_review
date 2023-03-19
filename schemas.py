from pydantic import BaseModel


class Review_out(BaseModel):
    text: str
    sentiment: str
    rate: int

class Review(BaseModel):
    text: str