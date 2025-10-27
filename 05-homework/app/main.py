from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class Input(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

class Output(BaseModel):
    converted_probability: float
    will_convert: bool

# Model
model_file = '/code/pipeline_v1.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(to_score: Input):

    # Parse input
    d = to_score.dict()

    # Prediction
    X = dv.transform([d])
    y_pred = model.predict_proba(X)[0, 1]
    converted = y_pred >= 0.5

    # Generates output
    result = Output(
        converted_probability=round(y_pred, 4),
        will_convert=converted
    )
    return result
