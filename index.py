import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

def robust_date_parser(val):
    try:
        if isinstance(val, pd.Timestamp):
            return val
        elif isinstance(val, (int, float)) and val > 40000:
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(val, unit="D")
        else:
            return pd.to_datetime(str(val), errors="coerce")
    except:
        return pd.NaT

st.title("Part Consumption Forecasting Tool")
st.write("Upload an Excel file with part usage history across multiple years or sheets.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    # Read all sheets and combine
    sheets = pd.read_excel(uploaded_file, sheet_name=None, header=1)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = df.columns.str.strip()
    
    # Parse dates and clean qty/part number
    df["Date of Initiation"] = df["Date of Initiation"].apply(robust_date_parser)
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df = df.dropna(subset=["Date of Initiation", "Qty", "Part Number"])
    df["Part Number"] = df["Part Number"].astype(str).str.strip()
    
    # Drop fully empty rows
    df = df.dropna(how="all")

    # Group by part and month
    df_grouped = df.groupby(["Part Number", pd.Grouper(key="Date of Initiation", freq="MS")])["Qty"].sum().reset_index()
    full_part_list = sorted(df_grouped["Part Number"].dropna().unique())

    # Filter to valid parts (must have at least one non-zero data point)
    valid_parts = []
    part_series_dict = {}
    for part in full_part_list:
        part_df = df_grouped[df_grouped["Part Number"] == part]
        if not part_df.empty and part_df["Qty"].sum() > 0:
            valid_parts.append(part)
            part_series_dict[part] = part_df

    forecast_rows = []
    forecast_headers = set()
    graph_images = {}

    for selected_part in valid_parts:
        try:
            part_df = part_series_dict[selected_part].copy()
            part_df.set_index("Date of Initiation", inplace=True)
            part_df = part_df.asfreq("MS", fill_value=0)
            series = part_df["Qty"]
            std_dev = np.std(series.values)
            forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=11, freq="MS")
            all_dates = series.index.tolist() + forecast_dates.tolist()

            use_arima = len(series) >= 12 and std_dev > 20

            if use_arima:
                model = ARIMA(series, order=(1, 1, 1))
                result = model.fit()
                forecast = result.get_forecast(steps=11)
                y_pred = np.concatenate([series.values, forecast.predicted_mean.values])
                se = forecast.se_mean.values
            else:
                X_train = np.arange(len(series)).reshape(-1, 1)
                X_forecast = np.arange(len(series) + 11).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X_train, series.values)
                y_pred = model.predict(X_forecast)
                y_train_pred = model.predict(X_train)
                mse = mean_squared_error(series.values, y_train_pred)
                base_std = np.sqrt(mse)
                se = np.array([base_std * (1 + 0.05 * i) for i in range(1, 12)])

            row_data = {"Part": selected_part}

            for i, dt in enumerate(all_dates):
                forecast_col = dt.strftime("%Y-%m") + " Forecast"
                row_data[forecast_col] = round(y_pred[i], 1)
                forecast_headers.add(forecast_col)

                for level, z in z_scores.items():
                    ci_col = dt.strftime("%Y-%m") + f" {level} CI Range"
                    idx = max(i - len(series), 0)
                    std = se[idx] if i >= len(series) else np.std(series.values)
                    lower = max(0, round(y_pred[i] - z * std, 1))
                    upper = round(y_pred[i] + z * std, 1)
                    row_data[ci_col] = f"{lower} - {upper}"
                    forecast_headers.add(ci_col)

            forecast_rows.append(row_data)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(series.index, series, label="Actual", marker='o')
            ax.plot(all_dates, y_pred, label="Forecast Line", linestyle='dotted', linewidth=2)

            forecast_start_idx = len(series)
            forecast_end_idx = len(y_pred)
            forecast_range = all_dates[forecast_start_idx:]

            for level, z in z_scores.items():
                lower_ci = []
                upper_ci = []
                for i in range(forecast_start_idx, forecast_end_idx):
                    idx = i - forecast_start_idx
                    lower = max(0, y_pred[i] - z * se[idx])
                    upper = y_pred[i] + z * se[idx]
                    lower_ci.append(lower)
                    upper_ci.append(upper)

                ax.fill_between(forecast_range, lower_ci, upper_ci, alpha=0.3, color=ci_colors[level], label=f"{level} CI")

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
        selected_part = st.selectbox("Select a part to view its forecast chart:", valid_parts)
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
