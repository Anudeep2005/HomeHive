import io
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")
print(features)

class HouseModel(BaseModel):
    MedInc : float = Field(gt = 0, description = "Median income of the neighbourhood")
    HouseAge : float = Field(gt = 0, description = "Average Age of the house in the block")
    AveRooms : float = Field(gt = 0, description = "Average rooms per house in the neighbourhood")
    AveBedrms : float = Field(gt = 0, description = "Average bedrooms per house in the neighbourhood")
    Population : float = Field(gt = 0, description = "Total population of the neighbourhood")
    AveOccup : float = Field(gt = 0, description = "Average occupants per house in the neighbourhood")
    Latitude : float = Field(ge = 32, le = 42, description = "Latitude Co-ordinates of the house")
    Longitude : float = Field(ge = -125, le = -114, description = "Longitude Co-ordinates of the house" )

@app.get("/")
def home():
    return {
        "message" : "California House Prediction API",
        "status" : "Running",
        "endpoint" : "Send POST request to /predict"
    }

@app.get("/health")
def health():
    return {
        "status" : "Running",
        "model" : "RandomForestRegressor",
        "features" : features,
        "avg error" : "$32,700"
    }

@app.post("/predict")
def predict_price(house : HouseModel):
    try :
        input = pd.DataFrame([{
            "MedInc" : house.MedInc,
            "HouseAge" : house.HouseAge,
            "AveRooms" : house.AveRooms,
            "AveBedrms" : house.AveBedrms,
            "Population" : house.Population,
            "AveOccup" : house.AveOccup,
            "Latitude" : house.Latitude,
            "Longitude" : house.Longitude
        }])
        predicted = model.predict(input[features])[0]
        predicted_usd_price = predicted * 100000
        return {
            "predicted_price" : f"${predicted_usd_price: ,.0f}",
            "predicted_price_margin" : f"{predicted:.2f}",
            "fidence_range" : f"${predicted_usd_price - 32700 :,.0f} to ${predicted_usd_price + 32700 : ,.0f}"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-file")
async def predict_file(file : UploadFile=File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code = 400,
            detail ="The file has to be of type CSV"
        )
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    required_columns = [
                        'MedInc',"HouseAge", 'AveRooms', 'AveBedrms',
                        'Population', 'AveOccup', 'Latitude', 'Longitude'
                        ]
    missing_columns = [
                        col for col in required_columns
                        if col not in df.columns
                      ]

    if missing_columns : 
        raise HTTPException(
            status_code = 400,
            detail = f"Missing columns : {missing_columns}"
        )
    if len(df) == 0:
        raise HTTPException(
            status_code = 400,
            detail = "Empty CSV File"
        )
    try:
        predictions = model.predict(df[required_columns])

        df["PredictedPrice"] = predictions * 100000

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )