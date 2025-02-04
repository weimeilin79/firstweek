import os
import weather
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/click", methods=["POST"])
def click():
    data = request.get_json()
    lat = data.get("lat")
    lng = data.get("lng")
    response = weather.find_location_info(lat, lng)
    return jsonify({"message": f"{response}"})

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
