import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Analytics")
st.title("🩺 Diabetes Progression Analytics")

db_data = load_diabetes()
X_tr, X_ts, y_tr, y_ts = train_test_split(db_data.data, db_data.target, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_tr, y_tr)
y_pred = reg_model.predict(X_ts)

m1, m2 = st.columns(2)
m1.metric("Mean Squared Error", round(mean_squared_error(y_ts, y_pred), 2))
m2.metric("R-Squared Score", round(r2_score(y_ts, y_pred), 2))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].scatter(y_ts, y_pred, color="#8e44ad", alpha=0.6)
ax[0].plot([y_ts.min(), y_ts.max()], [y_ts.min(), y_ts.max()], color="red", linestyle="--")
ax[0].set_title("Variance Analysis")

ax[1].scatter(X_ts[:, 2], y_pred, color="#27ae60", alpha=0.6)
ax[1].set_title("BMI Correlation")

st.pyplot(fig)
