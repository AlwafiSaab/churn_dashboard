import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Intelligence Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --- LOAD FILES ---
@st.cache_resource
def load_model_and_scaler():
    model = xgboost.XGBClassifier()
    model.load_model('xgb_model.json')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    X_test = joblib.load('X_test.pkl')
    df = joblib.load('df_with_scores.pkl')
    return X_test, df

model, scaler = load_model_and_scaler()
X_test, df = load_data()

# ================================

# SECTION 0 — CUSTOM DATA INPUT
# ================================
# upload their own customer file, and score it with the same model.
REQUIRED_COLUMNS = [
    'CreditScore', 'Geography', 'Gender', 'Age',
    'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

FEATURE_COLUMNS = list(X_test.columns)

# --- CSV TEMPLATE ---
template_data = {
    'CreditScore': [650],
    'Geography': ['France'],
    'Gender': ['Male'],
    'Age': [42],
    'Tenure': [5],
    'Balance': [125000],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [75000]
}

template_df = pd.DataFrame(template_data)


# --- PREPROCESS UPLOADED CSV ---
# This function converts the user's raw CSV into the exact feature
def preprocess_uploaded(df):
    df = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    gender_map = {
        'Male': 1,
        'Female': 0,
        'm': 1,
        'f': 0,
        '1': 1,
        '0': 0
    }
    normalized_gender = (
        df['Gender']
        .astype(str)
        .str.strip()
        .str.title()
    )

    invalid_gender = normalized_gender[~normalized_gender.isin(gender_map.keys())]
    if not invalid_gender.empty:
        raise ValueError(
            f"Unsupported Gender values found: {sorted(invalid_gender.unique())[:5]}. "
            "Please use Male or Female only."
        )

    df['Gender'] = normalized_gender.map(gender_map)

    geography_map = {
        'France': 'Geo_France',
        'Germany': 'Geo_Germany',
        'Spain': 'Geo_Spain'
    }
    normalized_geo = (
        df['Geography']
        .astype(str)
        .str.strip()
        .str.title()
    )
    unsupported_geo = normalized_geo[~normalized_geo.isin(geography_map.keys())]
    if not unsupported_geo.empty:
        raise ValueError(
            f"Unsupported Geography values found: {sorted(unsupported_geo.unique())[:5]}. "
            "Please use France, Germany, or Spain only."
        )

    for geo_name, flag_col in geography_map.items():
        df[flag_col] = (normalized_geo == geo_name).astype(int)

    df.drop(columns=['Geography'], inplace=True)

    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['HeavyUser'] = (df['NumOfProducts'] >= 3).astype(int)

    processed = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    processed = processed.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    return processed


# --- SCORE UPLOADED DATA ---
# This function applies the same scaler and model pipeline used by the
# dashboard to score any uploaded customer file.
def get_estimated_risk_metrics(_user_df, _threshold):
    processed = preprocess_uploaded(_user_df.copy())
    processed_scaled = scaler.transform(processed)
    churn_probability = model.predict_proba(processed_scaled)[:, 1]
    user_df = _user_df.copy()
    user_df['churn_probability'] = churn_probability
    high_risk = user_df[user_df['churn_probability'] > _threshold].copy()
    return user_df, high_risk


def calculate_revenue_at_risk(high_risk_count, clv, retention_cost, success_rate):
    net_customer_value = max(clv - retention_cost, 0)
    expected_recovery_rate = success_rate / 100
    return high_risk_count * net_customer_value * expected_recovery_rate

# --- CALCULATE RISK ---
X_test_scaled = scaler.transform(X_test)
df_test = df.loc[X_test.index].copy()
df_test['churn_probability'] = model.predict_proba(X_test_scaled)[:, 1]

st.title("🏦 Bank Churn Intelligence Dashboard")
st.caption("Identifying at-risk customers before they leave")
st.divider()

with st.sidebar:
    st.header("🎛️ Controls")
    threshold = st.slider("Risk Threshold", 0.5, 0.95, 0.7, step=0.01)
    geo = st.multiselect(
        "Filter by Geography",
        options=sorted(df_test['Geography'].dropna().unique())
    )

    st.divider()
    st.subheader("💼 Business Assumptions")
    clv = st.number_input(
        "Average Customer Lifetime Value ($)",
        min_value=100,
        max_value=50000,
        value=1200,
        step=100
    )
    retention_cost = st.number_input(
        "Retention Cost per Customer ($)",
        min_value=10,
        max_value=5000,
        value=200,
        step=10
    )
    success_rate = st.slider(
        "Retention Success Rate (%)",
        min_value=10,
        max_value=90,
        value=60
    )

    st.divider()
    st.subheader("📂 Predict Your Own Data")
    st.caption("Download the template, then upload your customer CSV to score it instantly.")
    st.download_button(
        label="⬇️ Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="churn_template.csv",
        mime="text/csv"
    )
    uploaded_file = st.file_uploader(
        "Upload your customer CSV file",
        type=["csv"],
        help="Expected columns: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary"
    )

high_risk = df_test[df_test['churn_probability'] > threshold].copy()
if geo:
    high_risk = high_risk[high_risk['Geography'].isin(geo)]

revenue_at_risk = calculate_revenue_at_risk(
    len(high_risk),
    clv,
    retention_cost,
    success_rate
)
active_high_risk = high_risk.copy()
active_revenue_at_risk = revenue_at_risk
active_at_risk_rate = round((len(high_risk) / len(df_test)) * 100, 1)

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        missing = [c for c in REQUIRED_COLUMNS if c not in user_df.columns]

        if missing:
            st.error(f"❌ Missing columns: {missing}")
            st.info("Please download the template above and fill it correctly.")
        else:
            user_df, user_high_risk = get_estimated_risk_metrics(user_df, threshold)
            st.success(f"✅ File accepted — {len(user_df):,} customers loaded")

            active_high_risk = user_high_risk.copy()
            active_revenue_at_risk = calculate_revenue_at_risk(
                len(user_high_risk),
                clv,
                retention_cost,
                success_rate
            )
            active_at_risk_rate = round((len(user_high_risk) / len(user_df)) * 100, 1)

            st.caption("📌 Dashboard metrics are now based on the uploaded customer dataset.")
    except Exception as exc:
        st.error(f"❌ Unable to process the uploaded file: {exc}")
        st.info("Please make sure your CSV follows the template columns and values exactly.")

# ================================
# SECTION 1 — KEY NUMBERS (TOP)
# ================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Revenue at Risk", f"${active_revenue_at_risk:,}")
with col2:
    st.metric("⚠️ High Risk Customers", f"{len(active_high_risk)}")
with col3:
    st.metric("📊 At-Risk Rate", f"{active_at_risk_rate}%")
st.info(f"→ **Recommended action:** Prioritize outreach to these {len(active_high_risk):,} customers this month.")

# ================================
# SECTION 2 — HIGH RISK TABLE
# ================================
st.subheader("⚠️ High Risk Customers")
st.caption(f"Churn probability above {threshold * 100:.0f}% — act on these first")

cols_to_show = [
    col for col in
    ['Age', 'NumOfProducts', 'IsActiveMember',
     'Balance', 'Tenure', 'churn_probability']
    if col in active_high_risk.columns
]

display = active_high_risk[cols_to_show].copy()
display['churn_probability'] = (
    display['churn_probability'] * 100
).round(1).astype(str) + '%'

display = display.rename(columns={
    'churn_probability': 'Churn Risk %'
})

st.dataframe(display.head(25), width='stretch')

st.download_button(
    "📥 Export High Risk List",
    active_high_risk.to_csv(index=False),
    "high_risk_customers.csv",
    mime="text/csv",
    key="download_high_risk_main"
)

# ================================
# SECTION 3 — SHAP CHART
# ================================
st.subheader("🔍 Top Churn Drivers")
st.caption("Factors Most Associated with Churn Risk")
    

@st.cache_data
def compute_shap(_model, _X_test_scaled):
    explainer = shap.TreeExplainer(_model)
    return explainer.shap_values(_X_test_scaled)

shap_values = compute_shap(model, X_test_scaled)

with plt.style.context('dark_background'):
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        max_display=8,
        show=False
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)