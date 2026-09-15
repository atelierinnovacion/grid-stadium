import json
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Character-to-Seat Projection Mapper")

st.title("Audiovisual Projection Mapper: Character-to-Seat Mapper")
st.write(
    "Upload text and a stadium grid mask. Each letter, number, symbol, and space will be assigned to a distinct seat shape."
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
            "WELCOME TO THE GRAND STADIUM LIGHT SHOW!",
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
        help="Increase if seats merge into a single shape.",
    )

if raw_text and grid_file:
    # 1. Parse text into individual characters (including spaces)
    characters = list(raw_text)
    num_chars = len(characters)

    # 2. Load grid mask
    grid_bytes = np.asarray(bytearray(grid_file.read()), dtype=np.uint8)
    grid_img = cv2.imdecode(grid_bytes, cv2.IMREAD_GRAYSCALE)
    grid_h, grid_w = grid_img.shape[:2]

    if invert_grid:
        grid_img = cv2.bitwise_not(grid_img)

    _, binary_mask = cv2.threshold(grid_img, 127, 255, cv2.THRESH_BINARY)

    # Separate connected seats
    if separation_strength > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (separation_strength * 2 + 1, separation_strength * 2 + 1),
        )
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

    raw_contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter invalid contours
    canvas_area = grid_h * grid_w
    valid_contours = []
    for cnt in raw_contours:
        area = cv2.contourArea(cnt)
        if min_seat_size <= area <= (canvas_area * 0.85):
            valid_contours.append(cnt)

    num_seats = len(valid_contours)

    if num_seats == 0:
        st.error(
            "⚠️ 0 seats detected! Enable 'Invert Colors' or lower 'Min Seat Size'."
        )
    elif num_seats == 1:
        st.warning(
            "⚠️ Only 1 seat detected! Increase 'Seat Separation Force (Erosion)'."
        )

    st.success(
        f"Detected **{num_seats}** seats for **{num_chars}** characters (letters, numbers, and spaces)."
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
        "total_characters_mapped": min(num_chars, num_seats),
        "seats": [],
    }

    for idx, (cnt, (x, y, bw, bh)) in enumerate(seat_data):
        seat_id = f"Seat_{idx+1}"
        center_x = int(x + bw / 2)
        center_y = int(y + bh / 2)

        assigned_char = characters[idx] if idx < num_chars else ""

        # Draw white background for active seats
        cv2.drawContours(output_canvas, [cnt], -1, (255, 255, 255), -1)

        # Draw character inside the seat (if not a space or empty)
        if assigned_char and assigned_char != " ":
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1

            (text_w, text_h), _ = cv2.getTextSize(
                assigned_char, font, font_scale, thickness
            )

            # Scale font to fit bounding box
            if text_w > (bw - 2) and bw > 4:
                font_scale = max(0.12, font_scale * ((bw - 2) / text_w))
                (text_w, text_h), _ = cv2.getTextSize(
                    assigned_char, font, font_scale, thickness
                )

            text_x = max(x, center_x - int(text_w / 2))
            text_y = max(y + text_h, center_y + int(text_h / 2))

            cv2.putText(
                output_canvas,
                assigned_char,
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
                "assigned_character": assigned_char,
                "is_space": assigned_char == " ",
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
        display_img,
        caption="Character-to-Seat Render",
        use_container_width=True,
    )

    st.subheader("JSON Character Mapping Output")
    json_str = json.dumps(json_mapping, indent=4)
    st.download_button(
        label="📄 Download JSON Coordinates",
        data=json_str,
        file_name="seat_char_mapping.json",
        mime="application/json",
    )
