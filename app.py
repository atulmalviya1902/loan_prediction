import streamlit as st
import pandas as pd
import joblib

# page configuration
st.set_page_config(
    page_title="loan prediction system",
    page_icon="🏦",
    layout="wide"
)

# custom dark interface
st.markdown("""
<style>

    /* main background */
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
    }

    /* headings */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }

    p, label {
        color: #cbd5e1;
    }

    /* sidebar */
    [data-testid="stSidebar"] {
        background-color: #111c30;
        border-right: 1px solid #263852;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* metric cards */
    [data-testid="stMetric"] {
        background-color: #172238;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #263852;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
    }

    /* buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        border: 1px solid #38bdf8;
    }

    /* input fields */
    div[data-baseweb="input"] input {
        background-color: #172238;
        color: #f1f5f9;
    }

    div[data-baseweb="select"] > div {
        background-color: #172238;
        color: #f1f5f9;
        border-color: #334155;
    }

    /* number input */
    [data-testid="stNumberInput"] {
        background-color: #172238;
    }

    /* form container */
    [data-testid="stForm"] {
        background-color: #111c30;
        padding: 20px;
        border: 1px solid #263852;
        border-radius: 12px;
    }

    /* divider */
    hr {
        border-color: #263852;
    }

    /* dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #263852;
        border-radius: 10px;
    }

    /* captions */
    .stCaption {
        color: #94a3b8;
    }

</style>
""", unsafe_allow_html=True)


# load dataset and model
df = pd.read_csv("loandata.csv")
model = joblib.load("loan_prediction_model.pkl")


# sidebar navigation
st.sidebar.title("🏦 loan prediction system")

page = st.sidebar.radio(
    "navigation",
    ["overview", "loan prediction", "data analysis"]
)

st.sidebar.markdown("---")
st.sidebar.caption("machine learning project")
st.sidebar.caption("random forest | model 1")


# --------------------------------------------------
# overview page
# --------------------------------------------------

if page == "overview":

    st.title("loan analytics overview")

    st.caption("Explore loan applications and approval trends.")

    total = len(df)

    approved = (df["Loan_Status"] == "Y").sum()

    rejected = (df["Loan_Status"] == "N").sum()

    approval_rate = (approved / total) * 100

    # kpi cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("total applications", total)

    col2.metric("approved loans", approved)

    col3.metric("rejected loans", rejected)

    col4.metric("approval rate", f"{approval_rate:.2f}%")

    st.markdown("---")

    # charts
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("loan status distribution")

        status_data = df["Loan_Status"].value_counts()

        status_data.index = [
            "approved" if x == "Y" else "rejected"
            for x in status_data.index
        ]

        st.bar_chart(status_data)

    with col2:

        st.subheader("property area distribution")

        area_data = df["Property_Area"].value_counts()

        st.bar_chart(area_data)

    st.markdown("---")

    st.subheader("recent loan applications")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# loan prediction page
# --------------------------------------------------

elif page == "loan prediction":

    st.title("loan approval prediction")

    st.caption(
        "Enter applicant information to predict loan approval."
    )

    with st.form("loan_form"):

        col1, col2 = st.columns(2)

        # personal information
        with col1:

            st.subheader("personal information")

            gender = st.selectbox(
                "gender",
                ["Male", "Female"]
            )

            married = st.selectbox(
                "married",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "dependents",
                ["0", "1", "2", "3+"]
            )

            education = st.selectbox(
                "education",
                ["Graduate", "Not Graduate"]
            )

            self_employed = st.selectbox(
                "self employed",
                ["Yes", "No"]
            )

        # financial information
        with col2:

            st.subheader("financial information")

            applicant_income = st.number_input(
                "applicant income",
                min_value=0,
                value=5000
            )

            coapplicant_income = st.number_input(
                "coapplicant income",
                min_value=0,
                value=0
            )

            loan_amount = st.number_input(
                "loan amount",
                min_value=0,
                value=128
            )

            loan_term = st.number_input(
                "loan term",
                min_value=1,
                value=360
            )

            credit_history = st.selectbox(
                "credit history",
                [1.0, 0.0],
                format_func=lambda x: "good" if x == 1.0 else "poor"
            )

            property_area = st.selectbox(
                "property area",
                ["Rural", "Semiurban", "Urban"]
            )

        predict_button = st.form_submit_button(
            "predict loan status",
            use_container_width=True
        )

    # prediction
    if predict_button:

        input_data = {
            "Gender": 1 if gender == "Male" else 0,

            "Married": 1 if married == "Yes" else 0,

            "Dependents": {
                "0": 0,
                "1": 1,
                "2": 2,
                "3+": 3
            }[dependents],

            "Education": 1 if education == "Graduate" else 0,

            "Self_Employed": 1 if self_employed == "Yes" else 0,

            "ApplicantIncome": applicant_income,

            "CoapplicantIncome": coapplicant_income,

            "LoanAmount": loan_amount,

            "Loan_Amount_Term": loan_term,

            "Credit_History": credit_history,

            "Property_Area_Rural": 1 if property_area == "Rural" else 0,

            "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,

            "Property_Area_Urban": 1 if property_area == "Urban" else 0
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=model.feature_names_in_
        )

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        approved_probability = (
            probability[list(model.classes_).index(1)] * 100
        )

        st.markdown("---")

        st.subheader("prediction result")

        if prediction == 1:

            st.success("loan approved!")

        else:

            st.error("loan rejected!")

        st.write("model approval probability")

        st.progress(
            float(approved_probability / 100)
        )

        st.metric(
            "approval probability",
            f"{approved_probability:.2f}%"
        )

        st.caption(
            "This is a machine learning prediction, "
            "not an actual bank decision."
        )


# --------------------------------------------------
# data analysis page
# --------------------------------------------------

elif page == "data analysis":

    st.title("interactive data analysis")

    st.caption(
        "Filter loan applications and explore different patterns."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_gender = st.selectbox(
            "select gender",
            ["All"] + sorted(
                df["Gender"].dropna().unique().tolist()
            )
        )

    with col2:

        selected_education = st.selectbox(
            "select education",
            ["All"] + sorted(
                df["Education"].dropna().unique().tolist()
            )
        )

    with col3:

        selected_area = st.selectbox(
            "select property area",
            ["All"] + sorted(
                df["Property_Area"].dropna().unique().tolist()
            )
        )

    # apply filters
    filtered_df = df.copy()

    if selected_gender != "All":

        filtered_df = filtered_df[
            filtered_df["Gender"] == selected_gender
        ]

    if selected_education != "All":

        filtered_df = filtered_df[
            filtered_df["Education"] == selected_education
        ]

    if selected_area != "All":

        filtered_df = filtered_df[
            filtered_df["Property_Area"] == selected_area
        ]

    st.markdown("---")

    st.metric(
        "filtered applications",
        len(filtered_df)
    )

    if not filtered_df.empty:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("loan status")

            st.bar_chart(
                filtered_df["Loan_Status"].value_counts()
            )

        with col2:

            st.subheader("applicant income")

            st.bar_chart(
                filtered_df["ApplicantIncome"]
            )

        st.markdown("---")

        st.subheader("filtered dataset")

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No applications found for these filters."
        )