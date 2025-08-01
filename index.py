import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

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

def is_stationary(series):
    try:
        result = adfuller(series.dropna())
        return result[1] < 0.05
    except:
        return False

def should_use_arima(series):
    return (
        len(series) >= 18
        and series.nunique() > 4
        and (series != series.mean()).sum() > 5
        and series.diff().abs().mean() > 0.5
        and is_stationary(series)
        and series.std() > 3  # new condition to ensure strong signal
    )

st.title("Part Consumption Forecasting Tool")
st.write("Upload an Excel file with part usage history across multiple years or sheets. There must be 3 columns titled Part, Date, and Quantity. These column headers must be in the very first row of your excel sheet")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

@st.cache_resource(show_spinner=False)
def process_forecast(uploaded_file):
    sheets = pd.read_excel(uploaded_file, sheet_name=None, header=0)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = df.columns.str.strip()

    col_map = {}
    qty_col = part_col = date_col = None

    for col in df.columns:
        col_clean = col.strip().lower()
        if qty_col is None and col_clean in ["qty", "quantity"]:
            qty_col = col
            col_map[col] = "Qty"
        elif part_col is None and col_clean in ["part", "part number"]:
            part_col = col
            col_map[col] = "Part Number"
        elif date_col is None and col_clean in ["date", "date of initiation"]:
            date_col = col
            col_map[col] = "Date of Initiation"

    missing = []
    if "Qty" not in col_map.values():
        missing.append("Qty or Quantity")
    if "Part Number" not in col_map.values():
        missing.append("Part or Part Number")
    if "Date of Initiation" not in col_map.values():
        missing.append("Date or Date of Initiation")

    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        return [], [], [], {}

    df.rename(columns=col_map, inplace=True)

    df["Date of Initiation"] = df["Date of Initiation"].apply(robust_date_parser)
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df = df.dropna(subset=["Date of Initiation", "Qty", "Part Number"])
    df["Part Number"] = df["Part Number"].astype(str).str.strip()
    df = df.dropna(how="all")

    df_grouped = df.groupby(["Part Number", pd.Grouper(key="Date of Initiation", freq="MS")])["Qty"].sum().reset_index()
    full_part_list = sorted(df_grouped["Part Number"].dropna().unique())

    valid_parts = []
    part_series_dict = {}
    for part in full_part_list:
        part_df = df_grouped[df_grouped["Part Number"] == part]
        if not part_df.empty and part_df["Qty"].sum() != 0:
            valid_parts.append(part)
            part_series_dict[part] = part_df

    forecast_rows = []
    forecast_headers = set()
    graph_images = {}

    for selected_part in valid_parts:
        try:
            part_df = part_series_dict[selected_part].copy()
            part_df.set_index("Date of Initiation", inplace=True)

            today = pd.Timestamp.today().replace(day=1)
            full_index = pd.date_range(start=part_df.index.min(), end=today, freq="MS")
            part_df = part_df.reindex(full_index, fill_value=0)

            series = part_df["Qty"]

            if series.isnull().any():
                raise ValueError("Input y contains NaN.")

            if (series != 0).sum() < 4:
                continue

            forecast_horizon = 30
            forecast_dates = pd.date_range(start=today, periods=forecast_horizon, freq="MS")
            all_dates = series.index.tolist() + forecast_dates.tolist()

            use_arima = should_use_arima(series)

            if use_arima:
                model = ARIMA(series, order=(10, 1, 3))
                result = model.fit()
                forecast = result.get_forecast(steps=forecast_horizon)
                y_pred = np.concatenate([series.values, forecast.predicted_mean.values])
                se = forecast.se_mean.values
                y_pred = np.clip(y_pred, 0, None)

                
            else:
                X_train = np.arange(len(series)).reshape(-1, 1)
                X_forecast = np.arange(len(series) + forecast_horizon).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X_train, series.values)
                y_pred = model.predict(X_forecast)
                y_train_pred = model.predict(X_train)
                mse = mean_squared_error(series.values, y_train_pred)
                base_std = np.sqrt(mse)
                se = np.array([base_std * (1 + 0.05 * i) for i in range(1, forecast_horizon + 1)])

            row_data = {"Part": selected_part}
            for i, dt in enumerate(all_dates):
                if dt >= forecast_dates[0]:
                    forecast_col = dt.strftime("%Y-%m") + " Forecast"
                    row_data[forecast_col] = round(y_pred[i], 1)
                    forecast_headers.add(forecast_col)

            forecast_rows.append(row_data)

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
                    lower = y_pred[i] - z * se[idx]
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
            st.warning(f"Part '{ selected_part}' skipped due to error: {e}")

    return valid_parts, forecast_headers, forecast_rows, graph_images

if uploaded_file:
    valid_parts, forecast_headers, forecast_rows, graph_images = process_forecast(uploaded_file)

    if forecast_rows:
        selected_part = st.selectbox("Select a part to view its forecast chart:", valid_parts)
        if selected_part in graph_images:
            st.image(graph_images[selected_part])
        else:
            st.info("Not enough data to forecast this part")

        results_df = pd.DataFrame(forecast_rows)
        results_df = results_df[["Part"] + sorted(c for c in results_df.columns if c != "Part")]

        # Reshape from wide to long format
        long_df = results_df.melt(id_vars=["Part"], var_name="Date", value_name="Forecast")
        long_df["Date"] = long_df["Date"].str.replace(" Forecast", "")  # Remove " Forecast" suffix
        long_df["Date"] = pd.to_datetime(long_df["Date"], format="%Y-%m")  # Convert to datetime

        long_df = long_df.sort_values(by=["Part", "Date"])

        with BytesIO() as excel_buf:
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                long_df.to_excel(writer, sheet_name="Forecast", index=False)
            excel_buf.seek(0)
            b64 = base64.b64encode(excel_buf.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_results.xlsx">Download Forecast Data (Excel)</a>'
            st.markdown(href, unsafe_allow_html=True)


        st.success("Forecasting complete. View the chart above and download the data below.")
