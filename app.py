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

        # cat_cols = []
        # for i in df.columns:
        #     if df[i].dtype == "object":
        #         cat_cols.append(i)
        #
        # num_cols = []
        # for i in df.columns:
        #     if df[i].dtype != "object" and i not in ["PROSPECTID", "Approved_Flag"]:
        #         num_cols.append(i)
        #
        # # Checking multicollinearity - calculating VIF (sequential)
        # vif_data = df[num_cols]
        # total_columns = vif_data.shape[1]
        # columns_to_be_kept = []
        # column_index = 0
        #
        # for i in range(0, total_columns):
        #
        #     vif_value = variance_inflation_factor(vif_data, column_index)
        #
        #     if vif_value <= 6:
        #         columns_to_be_kept.append(num_cols[i])
        #         column_index = column_index + 1
        #
        #     else:
        #         vif_data = vif_data.drop(columns=[num_cols[i]])
        #
        #
        # df = df[cat_cols + columns_to_be_kept]

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
