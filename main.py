import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile
import pandas as pd
import numpy as np
import sys
import os

# This is to make the utils file, which holds the custom processing class, FeatureSelector, to be available. 
# Serialised models usually require that any custom processing used in training is available at inference.
sys.path.append(os.path.abspath('src'))

# loading the model
model = joblib.load('src/model/model_v1.joblib')

# Instantiating a fastapi app
app = FastAPI()

# Creating a service with two enspoints- one for single requests and one for bulk requests

@app.get("/v1/check_house_price/single_prediction")
async def single_prediction(
    ID: int, loc: str, title: str, bedroom: int, bathroom: int, parking_space: float
):
    column_names = ['loc', 'title', 'bedroom', 'bathroom', 'parking_space']
    data = pd.DataFrame([[loc, title, bedroom, bathroom, parking_space]], columns=column_names)
    
    try:
        result = model.predict(data)
        result_value = result[0] if isinstance(result, np.ndarray) else result
        return {
            "ID": ID,
            'house_price': round(result_value, 3)
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting house price: {str(e)}")

@app.post("/v1/check_house_prices/bulk_prediction")
async def bulk_prediction(file: UploadFile = File(...)):
    try:
        # Checking which file extension so as to read the file accordingly
        file_extension = os.path.splitext(file.filename)[-1].lower()

        if file_extension == ".csv":
            df = pd.read_csv(file.file)
        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV and Excel files are allowed.")

        # Needed columns
        required_columns = ['ID', 'loc', 'title', 'bedroom', 'bathroom', 'parking_space']
        for column in required_columns:
            if column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing column: {column}")

        # Making house price predictions
        predictions = model.predict(df[required_columns[1:]])  # Use all columns except 'ID'

        # Converting predictions to a list of dictionaries
        results = [{
            "ID": id_,
            'house_price': round(pred, 3)
        } for id_, pred in zip(df['ID'], predictions)]

        return {'predictions': results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)