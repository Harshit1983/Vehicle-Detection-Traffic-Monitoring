import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import time


st.set_page_config(
    page_title="Smart Traffic Monitoring System",
    layout="wide"
)

st.title("🚦 Vehicle Detection & Traffic Monitoring")

st.markdown("""
This project uses:
- YOLOv8 for vehicle detection
- OpenCV for video processing
- Streamlit for dashboard visualization
""")


@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()


vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:


    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    line_y = height // 2

    vehicle_count = 0

    counted_ids = set()

    class_counts = defaultdict(int)

    traffic_history = []


    frame_placeholder = st.empty()

    col1, col2, col3 = st.columns(3)

    total_placeholder = col1.empty()
    traffic_placeholder = col2.empty()
    vehicle_placeholder = col3.empty()

    chart_placeholder = st.empty()


    stats_placeholder = st.sidebar.empty()


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break


        frame = cv2.resize(frame, (900, 500))


        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        boxes = results[0].boxes

        current_vehicles = 0

        if boxes is not None and boxes.id is not None:

            for box, track_id, cls in zip(
                    boxes.xyxy,
                    boxes.id,
                    boxes.cls):

                cls = int(cls)
                track_id = int(track_id)

                if cls in vehicle_classes:

                    current_vehicles += 1

                    x1, y1, x2, y2 = map(int, box)

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    vehicle_name = vehicle_classes[cls]

                    if line_y - 10 < cy < line_y + 10:

                        if track_id not in counted_ids:

                            counted_ids.add(track_id)

                            vehicle_count += 1

                            class_counts[vehicle_name] += 1


                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


                    label = f"{vehicle_name} ID:{track_id}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        4,
                        (0, 0, 255),
                        -1
                    )

        if current_vehicles <= 5:
            traffic_status = "Low Traffic"

        elif current_vehicles <= 15:
            traffic_status = "Medium Traffic"

        else:
            traffic_status = "Heavy Traffic"

        traffic_history.append(current_vehicles)


        cv2.line(
            frame,
            (0, line_y),
            (900, line_y),
            (255, 0, 0),
            3
        )


        total_placeholder.metric(
            "Total Vehicles Counted",
            vehicle_count
        )

        traffic_placeholder.metric(
            "Traffic Status",
            traffic_status
        )

        vehicle_placeholder.metric(
            "Vehicles In Current Frame",
            current_vehicles
        )


        stats_text = "## 🚗 Vehicle Statistics\n\n"

        for vehicle, count in class_counts.items():
            stats_text += f"{vehicle}: {count}\n"

        stats_placeholder.markdown(stats_text)


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )


        if len(traffic_history) > 5:

            chart_data = pd.DataFrame({
                "Vehicles": traffic_history
            })

            chart_placeholder.line_chart(chart_data)


    cap.release()


    st.success("✅ Video Processing Completed")

    st.markdown("## 📊 Final Traffic Report")

    report_df = pd.DataFrame({
        "Vehicle Type": list(class_counts.keys()),
        "Count": list(class_counts.values())
    })

    st.dataframe(report_df)


    csv = report_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Report CSV",
        data=csv,
        file_name="traffic_report.csv",
        mime="text/csv"
    )





