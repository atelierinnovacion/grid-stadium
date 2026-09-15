import json
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Text Projection Mapper")

st.title("Audiovisual Projection Mapper: Text-to-Seat Mapper")
st.write(
    "Upload a text file (or type text) and a binary grid mask. Each word will be mapped sequentially onto a seat shape."
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
        "Upload Grid Mask (White seats on Black background)",
        type=["jpg", "png", "jpeg"],
    )

if raw_text and grid_file:
    # Extract clean words list
    words = [w.strip() for w in raw_text.split() if w.strip()]

    # Read grid mask
    grid_bytes = np.asarray(bytearray(grid_file.read()), dtype=np.uint8)
    grid_img = cv2.imdecode(grid_bytes, cv2.IMREAD_GRAYSCALE)

    # Threshold grid to binary (255 = seat area, 0 = background)
    _, binary_mask = cv2.threshold(grid_img, 127, 255, cv2.THRESH_BINARY)

    # Find seat contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    num_seats = len(contours)

    if num_seats == 0:
        st.error("No valid seat contours found in the grid mask!")
    else:
        st.info(
            f"Detected **{num_seats}** seats and **{len(words)}** words available."
        )

        # Sort seats spatially (Top-to-Bottom, Left-to-Right)
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        seat_data = list(zip(contours, bounding_boxes))
        seat_data.sort(key=lambda item: (item[1][1] // 30, item[1][0]))

        grid_h, grid_w = grid_img.shape[:2]
        output_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        overlay_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        json_mapping = {
            "grid_dimensions": {"width": grid_w, "height": grid_h},
            "total_seats": num_seats,
            "total_words_mapped": min(len(words), num_seats),
            "seats": [],
        }

        # Render seats with background fills and words
        for idx, (cnt, (x, y, bw, bh)) in enumerate(seat_data):
            seat_id = f"Seat_{idx+1}"
            center_x = int(x + bw / 2)
            center_y = int(y + bh / 2)

            # Get assigned word (if text is shorter than total seats, leave empty or loop)
            assigned_word = words[idx] if idx < len(words) else ""

            # Draw white seat background fill on canvas
            cv2.drawContours(output_canvas, [cnt], -1, (255, 255, 255), -1)

            # Draw text inside seat bounding box
            if assigned_word:
                # Dynamic text scaling based on seat size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                # Calculate text size to center it
                (text_w, text_h), baseline = cv2.getTextSize(
                    assigned_word, font, font_scale, thickness
                )

                # Auto-scale font if text is wider than the seat bounding box
                if text_w > bw - 4 and bw > 10:
                    font_scale = max(0.2, font_scale * ((bw - 4) / text_w))
                    (text_w, text_h), baseline = cv2.getTextSize(
                        assigned_word, font, font_scale, thickness
                    )

                text_x = max(x, center_x - int(text_w / 2))
                text_y = max(y + text_h, center_y + int(text_h / 2))

                # Render text in black inside the seat
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

            # Save coordinates and word mapping to JSON
            json_mapping["seats"].append(
                {
                    "id": seat_id,
                    "index": idx + 1,
                    "center": {"x": center_x, "y": center_y},
                    "bounding_box": {"x": x, "y": y, "width": bw, "height": bh},
                    "assigned_word": assigned_word,
                }
            )

        # Build diagnostic overlay version
        overlay_canvas = output_canvas.copy()
        for idx, (_, (x, y, bw, bh)) in enumerate(seat_data):
            cv2.rectangle(
                overlay_canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 1
            )

        # Streamlit Render
        st.subheader("Mapped Projection Preview")
        show_boxes = st.checkbox("Show Seat Bounding Boxes", value=False)

        display_img = cv2.cvtColor(
            overlay_canvas if show_boxes else output_canvas, cv2.COLOR_BGR2RGB
        )
        st.image(
            display_img,
            caption="Mapped Text Output Render",
            use_container_width=True,
        )

        st.subheader("JSON Text-to-Seat Mapping")
        json_str = json.dumps(json_mapping, indent=4)
        st.download_button(
            label="📄 Download JSON Text Mapping",
            data=json_str,
            file_name="seat_text_mapping.json",
            mime="application/json",
        )
