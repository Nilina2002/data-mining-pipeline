from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "model"
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
STATS_PATH = MODEL_DIR / "dataset_stats.json"


st.set_page_config(
    page_title="Customer Segment Predictor",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_assets():
    with MODEL_PATH.open("rb") as file_handle:
        model = pickle.load(file_handle)

    with LABEL_ENCODER_PATH.open("rb") as file_handle:
        label_encoder = pickle.load(file_handle)

    with STATS_PATH.open("r", encoding="utf-8") as file_handle:
        stats = json.load(file_handle)

    return model, label_encoder, stats


def clamp_value(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def prediction_label(segment: str) -> str:
    descriptions = {
        "Champions": "High-value, recent, and frequent buyers. Prioritize retention and VIP offers.",
        "Loyal Customers": "Regular repeat buyers. Encourage upsell, bundles, and loyalty rewards.",
        "Potential Loyalists": "Promising customers who need a small nudge to increase frequency or spend.",
        "Lost / Inactive": "Low-engagement customers. Use win-back campaigns or reactivation offers.",
        "At-Risk": "Customers who are drifting away. Focus on timely reminders and special incentives.",
    }
    return descriptions.get(segment, "Customer segment predicted by the model.")


def format_currency(value: float) -> str:
    return f"£{value:,.2f}"


def predict_segment(model, label_encoder, recency: float, frequency: float, monetary: float):
    input_frame = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"],
    )
    probabilities = model.predict_proba(input_frame)[0]
    predicted_index = int(probabilities.argmax())
    predicted_segment = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])

    probability_frame = pd.DataFrame(
        {
            "Segment": label_encoder.classes_,
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    return predicted_segment, confidence, probability_frame


def main():
    st.title("Customer Segment Predictor")
    st.write(
        "Use the trained Random Forest model from this project to predict a customer's segment from RFM values."
    )

    if not MODEL_PATH.exists() or not LABEL_ENCODER_PATH.exists() or not STATS_PATH.exists():
        st.error(
            "Model artifacts were not found in outputs/model. Run notebooks/save_model.py first to generate them."
        )
        st.stop()

    model, label_encoder, stats = load_assets()

    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.subheader("Enter customer RFM values")

        recency = st.slider(
            "Recency (days since last purchase)",
            min_value=1,
            max_value=int(stats["recency_max"]),
            value=int(clamp_value(45, stats["recency_min"], stats["recency_max"])),
        )
        frequency = st.slider(
            "Frequency (number of orders)",
            min_value=int(stats["frequency_min"]),
            max_value=int(stats["frequency_max"]),
            value=int(clamp_value(3, stats["frequency_min"], stats["frequency_max"])),
        )
        monetary = st.number_input(
            "Monetary (total spend in £)",
            min_value=float(stats["monetary_min"]),
            max_value=float(stats["monetary_max"]),
            value=float(clamp_value(250.0, stats["monetary_min"], stats["monetary_max"])),
            step=50.0,
        )

        predict_clicked = st.button("Predict segment", type="primary")

        if predict_clicked:
            predicted_segment, confidence, probability_frame = predict_segment(
                model, label_encoder, recency, frequency, monetary
            )

            st.success(f"Predicted segment: {predicted_segment}")
            st.metric("Model confidence", f"{confidence:.1%}")
            st.info(prediction_label(predicted_segment))

            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Recency", f"{recency} days")
            result_col2.metric("Frequency", str(frequency))
            result_col3.metric("Monetary", format_currency(monetary))

            st.subheader("Prediction probabilities")
            st.bar_chart(
                probability_frame.set_index("Segment")[["Probability"]],
                horizontal=True,
            )
            st.dataframe(
                probability_frame.assign(Probability=probability_frame["Probability"].map(lambda value: f"{value:.2%}")),
                hide_index=True,
                use_container_width=True,
            )

    with right_col:
        st.subheader("Model summary")
        st.write(f"Total customers used for training: {stats['total_customers']:,}")
        st.write("Segments available to the model:")
        for segment in label_encoder.classes_:
            st.write(f"- {segment}")

        st.subheader("Training ranges")
        range_frame = pd.DataFrame(
            {
                "Feature": ["Recency", "Frequency", "Monetary"],
                "Minimum": [stats["recency_min"], stats["frequency_min"], stats["monetary_min"]],
                "99th percentile": [stats["recency_max"], stats["frequency_max"], stats["monetary_max"]],
            }
        )
        st.dataframe(range_frame, hide_index=True, use_container_width=True)

        st.subheader("Segment distribution")
        segment_counts = pd.DataFrame(
            {
                "Segment": list(stats["segment_counts"].keys()),
                "Customers": list(stats["segment_counts"].values()),
            }
        ).sort_values("Customers", ascending=False)
        st.dataframe(segment_counts, hide_index=True, use_container_width=True)

        st.caption(
            "This app predicts the customer segment from the trained Random Forest model saved in outputs/model."
        )


if __name__ == "__main__":
    main()