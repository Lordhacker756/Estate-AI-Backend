from flask import Flask, request, jsonify
from pydantic import BaseModel
import pandas as pd
import joblib
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


class RequestFeatures(BaseModel):
    bedroom: float
    layout_type: int
    area: float
    furnish_type: int
    bathroom: float
    city: int
    agent: int
    builder: int
    owner: int
    apartment: int
    independent_floor: int
    independent_house: int
    penthouse: int
    studio_apartment: int
    villa: int


with open('model.joblib', 'rb') as f:
    model = joblib.load(f)


@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    req = RequestFeatures(**request.json)
    df = pd.DataFrame([req.dict()], columns=req.dict().keys())
    try:
        print("Values in payload", req.dict(), [list(req.dict().values())])
        res = model.predict([list(req.dict().values())])
        return jsonify({"prediction": res[0]})
    except Exception as e:
        return jsonify({"error": str(e)})
