import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
model_path = "model.pkl"
try:
  with open(model_path, "rb") as f:
    model = pickle.load(f)
except Exception as e:
  model = None
  print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen py-10 px-4">
    <div class="max-w-3xl mx-auto bg-white rounded-2xl shadow-xl overflow-hidden">
        <!-- Header -->
        <div class="bg-gradient-to-r from-blue-600 to-indigo-600 py-6 px-8 text-white text-center">
            <h1 class="text-3xl font-bold tracking-wide">Loan Eligibility Predictor</h1>
            <p class="text-blue-100 mt-2 text-sm">Enter your details below to check your loan approval status.</p>
        </div>

        <!-- Content Section -->
        <div class="p-8">
            {% if error %}
            <div class="mb-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-md">
                <p>{{ error }}</p>
            </div>
            {% endif %}

            {% if prediction is not none %}
            <div class="mb-8 p-6 rounded-xl text-center shadow-inner {% if prediction == 1 or prediction == 'Y' or prediction == 'Approved' %}bg-green-50 border border-green-200 text-green-800{% else %}bg-red-50 border border-red-200 text-red-800{% endif %}">
                <h2 class="text-2xl font-bold mb-1">
                    {% if prediction == 1 or prediction == 'Y' or prediction == 'Approved' %}
                    🎉 Congratulations! Your Loan is Approved.
                    {% else %}
                    ❌ We're sorry, your Loan Application is Rejected.
                    {% endif %}
                </h2>
                <p class="text-sm opacity-80 mt-1">Based on the financial and personal details provided.</p>
            </div>
            {% endif %}

            <form method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Gender -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Gender</label>
                    <select name="Gender" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>

                <!-- Married -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Marital Status</label>
                    <select name="Married" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <!-- Dependents -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Dependents</label>
                    <select name="Dependents" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white">
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3+</option>
                    </select>
                </div>

                <!-- Education -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Education</label>
                    <select name="Education" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white">
                        <option value="1">Graduate</option>
                        <option value="0">Not Graduate</option>
                    </select>
                </div>

                <!-- Self_Employed -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Self Employed</label>
                    <select name="Self_Employed" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <!-- ApplicantIncome -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Applicant Income ($)</label>
                    <input type="number" name="ApplicantIncome" placeholder="e.g., 5000" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition">
                </div>

                <!-- CoapplicantIncome -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Coapplicant Income ($)</label>
                    <input type="number" step="any" name="CoapplicantIncome" placeholder="e.g., 1500" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition">
                </div>

                <!-- LoanAmount -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Loan Amount (in thousands)</label>
                    <input type="number" step="any" name="LoanAmount" placeholder="e.g., 120" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition">
                </div>

                <!-- Loan_Amount_Term -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Loan Amount Term (Months)</label>
                    <input type="number" name="Loan_Amount_Term" placeholder="e.g., 360" value="360" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition">
                </div>

                <!-- Credit_History -->
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Credit History</label>
                    <select name="Credit_History" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white">
                        <option value="1">All debts paid timely (1)</option>
                        <option value="0">Unpaid debts / Bad history (0)</option>
                    </select>
                </div>

                <!-- Property_Area -->
                <div class="md:col-span-2">
                    <label class="block text-sm font-semibold text-gray-700 mb-2">Property Area</label>
                    <select name="Property_Area" required class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition bg-white">
                        <option value="2">Urban</option>
                        <option value="1">Semiurban</option>
                        <option value="0">Rural</option>
                    </select>
                </div>

                <!-- Submit Button -->
                <div class="md:col-span-2 mt-4">
                    <button type="submit" class="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:from-blue-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition duration-200">
                        Predict Loan Eligibility
                    </button>
                </div>
            </form>
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  error = None

  if request.method == "POST":
    if model is None:
      error = (
          "Model file could not be loaded. Please ensure model.pkl is present."
      )
    else:
      try:
        features = [
            float(request.form["Gender"]),
            float(request.form["Married"]),
            float(request.form["Dependents"]),
            float(request.form["Education"]),
            float(request.form["Self_Employed"]),
            float(request.form["ApplicantIncome"]),
            float(request.form["CoapplicantIncome"]),
            float(request.form["LoanAmount"]),
            float(request.form["Loan_Amount_Term"]),
            float(request.form["Credit_History"]),
            float(request.form["Property_Area"]),
        ]
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
      except Exception as e:
        error = f"Error during prediction: {str(e)}"

  return render_template_string(
      HTML_TEMPLATE, prediction=prediction, error=error
  )


if __name__ == "__main__":
  app.run(debug=True)
