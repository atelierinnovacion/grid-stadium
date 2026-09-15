import json
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Text Projection Mapper")

st.title("Audiovisual Projection Mapper: Text-to-Seat Mapper")
st.write(
    "Upload your text and grid mask. Adjust the seat separation controls if seats are detection as one big shape."
)

col1, col2 = st.columns(2)
with col1:
    text_input_mode = st.radio(
        "Text Input Method", ["Type / Paste Text", "Upload .txt File"]
    )
    raw_text = ""
    if text_input_mode == "Type / Paste Text":
        raw_text = st.text_area(
            "Enter text content:",
            "WELCOME TO THE GRAND STADIUM LIGHT SHOW ENJOY THE EXPERIENCE",
            height=150,
        )
    else:
        text_file = st.file_uploader("Upload Text File", type=["txt"])
        if text_file:
            raw_text = text_file.read().decode("utf-8")

with col2:
    grid_file = st.file_uploader(
        "Upload Stadium Grid Mask",
        type=["jpg", "png", "jpeg"],
    )

    # Calibration controls to fix single-contour / merged-seat problems
    st.markdown("**Grid Processing Calibration**")
    invert_grid = st.checkbox(
        "Invert Colors (Check if seats are BLACK on WHITE background)",
        value=False,
    )
    min_seat_size = st.slider(
        "Min Seat Size (Pixels)",
        min_value=5,
        max_value=1000,
        value=20,
    )
    separation_strength = st.slider(
        "Seat Separation Force (Erosion)",
        min_value=0,
        max_value=15,
        value=2,
        help="Increase this value if all seats are merging into 1 shape.",
    )

if raw_text and grid_file:
    # 1. Clean words list
    words = [w.strip() for w in raw_text.split() if w.strip()]
    num_words = len(words)

    # 2. Load grid mask
    grid_bytes = np.asarray(bytearray(grid_file.read()), dtype=np.uint8)
    grid_img = cv2.imdecode(grid_bytes, cv2.IMREAD_GRAYSCALE)
    grid_h, grid_w = grid_img.shape[:2]

    # Invert colors if user selected black-on-white seats
    if invert_grid:
        grid_img = cv2.bitwise_not(grid_img)

    # Binary threshold
    _, binary_mask = cv2.threshold(grid_img, 127, 255, cv2.THRESH_BINARY)

    # Apply Morphological Erode to cut connecting lines between seats
    if separation_strength > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (separation_strength * 2 + 1, separation_strength * 2 + 1),
        )
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    # Find raw contours
    raw_contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out contours that are too small (noise) or too large (entire image frame)
    canvas_area = grid_h * grid_w
    valid_contours = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if min_seat_size <= area <= (canvas_area * 0.85):
            valid_contours.append(cnt)

    num_seats = len(valid_contours)

    if num_seats == 0:
        st.error(
            "⚠️ 0 seats detected! Enable 'Invert Colors' or decrease 'Min Seat Size'."
        )
    elif num_seats == 1:
        st.warning(
            "⚠️ Only 1 seat detected! Increase the 'Seat Separation Force (Erosion)' slider to force connected seat shapes apart."
        )

    st.success(
        f"Detected **{num_seats}** individual seats for **{num_words}** words."
    )

    # Sort seats spatially (Top-to-Bottom, Left-to-Right)
    row_threshold = max(10, grid_h // 25)
    seat_data = [(cnt, cv2.boundingRect(cnt)) for cnt in valid_contours]
    seat_data.sort(key=lambda item: (item[1][1] // row_threshold, item[1][0]))

    # --- DRAW MAPPED OUTPUT ---
    output_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    json_mapping = {
        "grid_dimensions": {"width": grid_w, "height": grid_h},
        "total_seats_detected": num_seats,
        "total_words_mapped": min(num_words, num_seats),
        "seats": [],
    }

    for idx, (cnt, (x, y, bw, bh)) in enumerate(seat_data):
        seat_id = f"Seat_{idx+1}"
        center_x = int(x + bw / 2)
        center_y = int(y + bh / 2)

        assigned_word = words[idx] if idx < num_words else ""

        # Fill seat background in white
        cv2.drawContours(output_canvas, [cnt], -1, (255, 255, 255), -1)

        # Draw word inside seat contour
        if assigned_word:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                assigned_word, font, font_scale, thickness
            )

            # Auto-scale font down if text is wider than the seat box
            if text_w > (bw - 2) and bw > 4:
                font_scale = max(0.12, font_scale * ((bw - 2) / text_w))
                (text_w, text_h), baseline = cv2.getTextSize(
                    assigned_word, font, font_scale, thickness
                )

            text_x = max(x, center_x - int(text_w / 2))
            text_y = max(y + text_h, center_y + int(text_h / 2))

            cv2.putText(
                output_canvas,
                assigned_word,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        json_mapping["seats"].append(
            {
                "id": seat_id,
                "index": idx + 1,
                "center": {"x": center_x, "y": center_y},
                "bounding_box": {"x": x, "y": y, "width": bw, "height": bh},
                "assigned_word": assigned_word,
            }
        )

    # Diagnostic Overlay
    overlay_canvas = output_canvas.copy()
    for idx, (_, (x, y, bw, bh)) in enumerate(seat_data):
        cv2.rectangle(overlay_canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

    st.subheader("Mapped Projection Preview")
    show_boxes = st.checkbox("Show Seat Bounding Boxes", value=True)

    display_img = cv2.cvtColor(
        overlay_canvas if show_boxes else output_canvas, cv2.COLOR_BGR2RGB
    )
    st.image(
        display_img, caption="Text-to-Seat Render", use_container_width=True
    )

    st.subheader("JSON Mapping Output")
    json_str = json.dumps(json_mapping, indent=4)
    st.download_button(
        label="📄 Download JSON Coordinates",
        data=json_str,
        file_name="seat_text_mapping.json",
        mime="application/json",
    )
