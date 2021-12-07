from flask import Flask, request

from flask_restful import Api
from flask_cors import CORS

from ModelService import ModelService

app = Flask(__name__)
api = Api(app)
CORS(app)


@app.route("/predict", methods=["POST"])
def process_image():
    model_service = ModelService()
    payload = request.get_json()
    img = payload['image']
    model_service = ModelService()
    prediction = model_service.predict(img)
    print(type(prediction))
    return {"msg": "success", "data": prediction.decode('utf8').replace("'", '"')}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)




