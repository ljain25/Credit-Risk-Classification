import streamlit as st
import pandas as pd
import pickle
from io import StringIO
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


# Streamlit UI
st.title("📊 Credit Risk Classification: Predicting Customer Risk Levels")
st.write("Upload a CSV file, and the model will make predictions!")

# File Upload
uploaded_file = st.file_uploader("  ", type=["csv"])

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(stringio)
    st.dataframe(df.head())
    df2 = pd.read_csv("training_data.csv")

    try:

        df.loc[df["EDUCATION"] == "SSC", ["EDUCATION"]] = 1
        df.loc[df["EDUCATION"] == "12TH", ["EDUCATION"]] = 2
        df.loc[df["EDUCATION"] == "GRADUATE", ["EDUCATION"]] = 3
        df.loc[df["EDUCATION"] == "UNDER GRADUATE", ["EDUCATION"]] = 3
        df.loc[df["EDUCATION"] == "POST-GRADUATE", ["EDUCATION"]] = 4
        df.loc[df["EDUCATION"] == "OTHERS", ["EDUCATION"]] = 1
        df.loc[df["EDUCATION"] == "PROFESSIONAL", ["EDUCATION"]] = 3

        df["EDUCATION"].value_counts()
        df["EDUCATION"].dtype
        df["EDUCATION"] = df["EDUCATION"].astype(int)
        df.info()

        df_encoded = pd.get_dummies(df, columns=["MARITALSTATUS", "GENDER", "last_prod_enq2", "first_prod_enq2"],
                                    dtype="int", drop_first=True)

        df_encoded = df_encoded[list(df2.columns)[1:]]

        filename = "model.sav"
        load_model = pickle.load(open(filename, "rb"))

        x = df2.iloc[:, 1:]
        y = df2.iloc[:, 0]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        load_model.fit(x, y_encoded)

        predictions = load_model.predict(df_encoded)
        df_encoded["Predictions"] = predictions + 1

        df_encoded["Predictions"] = "P" + df_encoded["Predictions"].astype(str)

        csv = df_encoded.to_csv(index = False).encode("utf-8")
        st.write("Result:")

        st.dataframe(df_encoded.head())

        st.download_button("Download Result", csv, "Risk_levels.csv", "text/csv")

    except Exception as e:
        st.error(f"Error in processing and prediction: {e}")
