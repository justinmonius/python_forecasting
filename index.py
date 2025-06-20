import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64

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
    months = df.iloc[1, 1:].astype(int).values
    dates = pd.to_datetime([f"{y}-{m:02d}" for y, m in zip(years, months)])

    forecast_results = {}
    graph_images = {}

    part_names = []

    for row in range(2, df.shape[0]):
        part_name = df.iloc[row, 0]
        values = df.iloc[row, 1:].astype(float).values

        # Skip rows with missing values
        if np.isnan(values).any():
            st.warning(f"⚠️ Skipped part '{part_name}' due to missing values.")
            continue

        X = np.arange(len(values)).reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, values)
        y_pred = model.predict(X)

        residuals = values - y_pred
        mse = np.mean(residuals**2)
        se = np.sqrt(mse)

        # Create forecast DataFrame
        forecast_row = {}
        forecast_row['Part'] = part_name

        for date_idx, dt in enumerate(dates):
            month_str = dt.strftime('%Y-%m')
            forecast_row[f'{month_str} Forecast'] = round(y_pred[date_idx], 1)
            for level, z in z_scores.items():
                ci = z * se
                ci_range = f"{round(y_pred[date_idx] - ci, 1)} - {round(y_pred[date_idx] + ci, 1)}"
                forecast_row[f'{month_str} {level} CI'] = ci_range

        forecast_results[part_name] = forecast_row
        part_names.append(part_name)

        # Plot chart
        fig, ax = plt.subplots(figsize=(10, 5))
        cutoff = pd.to_datetime("2025-05-31")
        actual_mask = dates <= cutoff

        ax.plot(dates[actual_mask], values[actual_mask], label="Actual", marker='o')
        ax.plot(dates, y_pred, label="Forecast Line", linestyle='dotted')

        for level, z in z_scores.items():
            ci = z * se
            upper = y_pred + ci
            lower = y_pred - ci
            ax.fill_between(dates, lower, upper, alpha=0.3, color=ci_colors[level], label=f"{level} CI")

        ax.set_title(f"Forecast for {part_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Quantity")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        # Save to image buffer
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        graph_images[part_name] = buf

    if forecast_results:
        selected_part = st.selectbox("Select a part to view its forecast chart:", sorted(part_names))
        if selected_part in graph_images:
            st.image(graph_images[selected_part])

        # Create downloadable Excel
        results_df = pd.DataFrame.from_dict(forecast_results, orient='index')
        with BytesIO() as excel_buf:
            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name="Forecast")
            excel_buf.seek(0)
            b64 = base64.b64encode(excel_buf.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_results.xlsx">Download Forecast Data (Excel)</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.success("Forecasting complete. Select a part above to view its chart and download the data below.")
