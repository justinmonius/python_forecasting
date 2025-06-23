import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# Z-scores and colors for confidence intervals
z_scores = {
    "95%": 1.96,
    "80%": 1.282,
    "50%": 0.674
}

ci_colors = {
    "95%": "lightblue",
    "80%": "skyblue",
    "50%": "deepskyblue"
}

st.title("Part Consumption Forecasting Tool")
st.write("Upload an Excel file with part numbers as rows and months as columns.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None)

    # Extract headers
    years = df.iloc[0, 1:].astype(int).values
    raw_months = df.iloc[1, 1:].values
    month_lookup = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    months = []
    for m in raw_months:
        if isinstance(m, str):
            key = m.strip().lower()
            if key in month_lookup:
                months.append(month_lookup[key])
            else:
                months.append(month_lookup[key[:3]])
        else:
            months.append(int(m))
    months = np.array(months)

    dates = pd.to_datetime([f"{y}-{m:02d}" for y, m in zip(years, months)])

    # Extend by 11 months
    last_date = dates[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=11, freq='MS')
    all_dates = dates.tolist() + forecast_dates.tolist()

    forecast_data = []
    part_names = []
    graph_images = {}

    for row in range(2, df.shape[0]):
        part_name = df.iloc[row, 0]
        values = df.iloc[row, 1:].astype(float).values

        if np.isnan(values).any():
            st.warning(f"⚠️ Skipped part '{part_name}' due to missing values.")
            continue

        series = pd.Series(values, index=dates)

        try:
            model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            result = model.fit(disp=False)
            forecast = result.get_forecast(steps=11)
            y_pred = np.concatenate([series.values, forecast.predicted_mean.values])
            se = forecast.se_mean.values
        except:
            st.warning(f"⚠️ Could not fit SARIMA model for part '{part_name}'.")
            continue

        row_data = [part_name]
        for i, dt in enumerate(all_dates):
            row_data.append(round(y_pred[i], 1))
            for level, z in z_scores.items():
                if i < len(series):
                    std_dev = np.std(series.values)
                    lower = round(y_pred[i] - z * std_dev, 1)
                    upper = round(y_pred[i] + z * std_dev, 1)
                else:
                    idx = i - len(series)
                    lower = round(y_pred[i] - z * se[idx], 1)
                    upper = round(y_pred[i] + z * se[idx], 1)
                row_data.append(f"{lower} - {upper}")

        forecast_data.append(row_data)
        part_names.append(part_name)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, series, label="Actual", marker='o')
        ax.plot(all_dates, y_pred, label="Forecast Line", linestyle='dotted', linewidth=2)

        for level, z in z_scores.items():
            upper_ci = y_pred.copy()
            lower_ci = y_pred.copy()
            for i in range(len(series), len(y_pred)):
                idx = i - len(series)
                lower_ci[i] = y_pred[i] - z * se[idx]
                upper_ci[i] = y_pred[i] + z * se[idx]
            ax.fill_between(all_dates, lower_ci, upper_ci, alpha=0.3, color=ci_colors[level], label=f"{level} CI")

        ax.set_title(f"Forecast for {part_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Quantity")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        graph_images[part_name] = buf

    if forecast_data:
        selected_part = st.selectbox("Select a part to view its forecast chart:", sorted(part_names))
        if selected_part in graph_images:
            st.image(graph_images[selected_part])

        headers = ["Part"]
        for dt in all_dates:
            headers.append(dt.strftime("%Y-%m") + " Forecast")
            for level in z_scores:
                headers.append(dt.strftime("%Y-%m") + f" {level} CI Range")

        results_df = pd.DataFrame(forecast_data, columns=headers)

        with BytesIO() as excel_buf:
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name="Forecast", index=False)
            excel_buf.seek(0)
            b64 = base64.b64encode(excel_buf.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_results.xlsx">Download Forecast Data (Excel)</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.success("Forecasting complete. Select a part above to view its chart and download the data below.")
