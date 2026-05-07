import pandas as pd


def predict(data: dict, model, preprocessing):
    df = pd.DataFrame([data])
    df_preprocessed = preprocessing.transform(df)
    prediction = model.predict(df_preprocessed)[0]
    return {"prediction": float(prediction)}
