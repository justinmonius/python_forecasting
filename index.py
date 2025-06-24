import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
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

def safe_excel_date(val):
    try:
        if isinstance(val, pd.Timestamp):
            return val
        elif isinstance(val, (int, float)):
            if val > 40000:
                return pd.to_datetime("1899-12-30") + pd.to_timedelta(val, unit="D")
            else:
                return pd.NaT
        else:
            return pd.to_datetime(val, errors="coerce")
    except:
        return pd.NaT

st.title("Part Consumption Forecasting Tool")
st.write("Upload an Excel file with part usage history OR raw date-quantity logs.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=1)
    df.columns = df.columns.str.strip()

    if "Date of Initiation" in df.columns and "Qty" in df.columns and "Part Number" in df.columns:
        df["Date of Initiation"] = df["Date of Initiation"].apply(safe_excel_date)
        df = df.dropna(subset=["Date of Initiation"])

        # Clean part numbers to avoid duplicates
        df["Part Number"] = df["Part Number"].astype(str).str.strip()

        df_grouped = df.groupby(["Part Number", pd.Grouper(key="Date of Initiation", freq="MS")])["Qty"].sum().reset_index()
        part_list = sorted([str(p) for p in df_grouped["Part Number"].unique()])

        forecast_rows = []
        forecast_headers = set()
        part_names = []
        graph_images = {}

        for selected_part in part_list:
            try:
                part_df = df_grouped[df_grouped["Part Number"] == selected_part].copy()
                if part_df.empty:
                    continue

                part_df.set_index("Date of Initiation", inplace=True)
                part_df = part_df.asfreq("MS", fill_value=0)

                series = part_df["Qty"]
                forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=11, freq="MS")
                all_dates = series.index.tolist() + forecast_dates.tolist()

                if len(series) >= 12:
                    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    result = model.fit(disp=False)
                    forecast = result.get_forecast(steps=11)
                    y_pred = np.concatenate([series.values, forecast.predicted_mean.values])
                    se = forecast.se_mean.values
                else:
                    X_train = np.arange(len(series)).reshape(-1, 1)
                    X_forecast = np.arange(len(series) + 11).reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X_train, series.values)
                    y_pred = model.predict(X_forecast)
                    se = np.full(11, np.std(series.values))

                row_data = {"Part": selected_part}

                for i, dt in enumerate(all_dates):
                    forecast_col = dt.strftime("%Y-%m") + " Forecast"
                    row_data[forecast_col] = round(y_pred[i], 1)
                    forecast_headers.add(forecast_col)

                    for level, z in z_scores.items():
                        ci_col = dt.strftime("%Y-%m") + f" {level} CI Range"
                        idx = max(i - len(series), 0)
                        std = se[idx] if i >= len(series) else np.std(series.values)
                        lower = round(y_pred[i] - z * std, 1)
                        upper = round(y_pred[i] + z * std, 1)
                        row_data[ci_col] = f"{lower} - {upper}"
                        forecast_headers.add(ci_col)

                forecast_rows.append(row_data)
                part_names.append(selected_part)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(series.index, series, label="Actual", marker='o')
                ax.plot(all_dates, y_pred, label="Forecast Line", linestyle='dotted', linewidth=2)

                for level, z in z_scores.items():
                    upper_ci = y_pred.copy()
                    lower_ci = y_pred.copy()
                    for i in range(len(series), len(y_pred)):
                        idx = i - len(series)
                        lower_ci[i] = y_pred[i] - z * se[idx]
                        upper_ci[i] = y_pred[i] + z * se[idx]
                    ax.fill_between(all_dates, lower_ci, upper_ci, alpha=0.3, color=ci_colors[level], label=f"{level} CI")

                ax.set_title(f"Forecast for {selected_part}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Quantity")
                ax.legend()
                ax.grid(True)
                fig.tight_layout()

                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                graph_images[selected_part] = buf

            except Exception as e:
                st.warning(f"Part '{selected_part}' skipped due to error: {e}")

        if forecast_rows:
            selected_part = st.selectbox("Select a part to view its forecast chart:", part_names)
            if selected_part in graph_images:
                st.image(graph_images[selected_part])

            all_headers = ["Part"] + sorted(forecast_headers)
            results_df = pd.DataFrame(forecast_rows).reindex(columns=all_headers)

            with BytesIO() as excel_buf:
                with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name="Forecast", index=False)
                excel_buf.seek(0)
                b64 = base64.b64encode(excel_buf.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_results.xlsx">Download Forecast Data (Excel)</a>'
                st.markdown(href, unsafe_allow_html=True)

            st.success("Forecasting complete. View the chart above and download the data below.")