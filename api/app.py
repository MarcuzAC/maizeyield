from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import logging

# Loading model (which already includes the preprocessor)
dtr = pickle.load(open('model.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collecting input features safely using get()
        district = request.form.get('district')
        average_rain_fall_mm_per_year = request.form.get('average_rain_fall_mm_per_year')
        fertilizer = request.form.get('fertilizer')
        avg_temp = request.form.get('avg_temp')
        Year = request.form.get('Year')
        
        print("Received values:")
        print("Year:", Year)
        print("Average Rainfall:", average_rain_fall_mm_per_year)
        print("Fertilizer:", fertilizer)
        print("Average Temperature:", avg_temp)
        print("District:", district)

        # Check for empty fields
        if not Year or not average_rain_fall_mm_per_year or not fertilizer or not avg_temp or not district:
            return render_template("index.html", error="All fields are required.")

        # Validate and convert inputs
        try:
            # Convert numeric inputs to appropriate types
            average_rain_fall_mm_per_year = float(average_rain_fall_mm_per_year)
            fertilizer = float(fertilizer)
            avg_temp = float(avg_temp)
            Year = int(Year)
        except ValueError:
            return render_template('index.html', error="Invalid input types. Please enter numeric values for numeric fields.")

        # Prepare input data for the model
        input_data = {
            'Year': Year,
            'average_rain_fall_mm_per_year': average_rain_fall_mm_per_year,
            'fertilizer': fertilizer,
            'avg_temp': avg_temp,
            'district': district
        }

        # Convert input_data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Check for any None values and replace them if necessary
        input_df.fillna('Unknown', inplace=True)  # Replace None with a placeholder, if applicable

        # Make prediction using your model
        prediction = dtr.predict(input_df)

        # Logging the input DataFrame and the prediction
        logging.debug(f"Input DataFrame: {input_df}")
        logging.debug(f"Prediction: {prediction}")

        # Pass input data to the template for display
        return render_template('index.html', prediction=prediction[0], input_data=input_data)

if __name__ == "__main__":
    app.run(debug=True)
