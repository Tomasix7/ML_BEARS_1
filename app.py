from flask import Flask, request, jsonify, render_template
from fastai.learner import load_learner
from fastai.vision.all import PILImage
import io

app = Flask(__name__)
learn_inf = load_learner('bears_model.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    img = PILImage.create(io.BytesIO(img_bytes))
    pred, pred_idx, probs = learn_inf.predict(img)

    return jsonify({
        'prediction': str(pred),
        'probability': f"{probs[pred_idx]:.04f}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
