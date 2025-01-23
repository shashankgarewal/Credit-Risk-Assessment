from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__) # flask instance

# route for homepage
@app.route('/')
def home():
    return "Welcome to Flask!"

# royute for predict
@app.route('/api/predict', methods=['GET', 'POST'])
def api_data():
    if request.method == 'POST':
        data = request.get_json()  # Get JSON data from the request
        return jsonify({"message": "Data received", "data": data})
    return jsonify({"message": "Send a POST request with JSON data"})

# Step 5: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
