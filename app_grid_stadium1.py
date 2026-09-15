import json
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Projection Mapping Seat Divider")

st.title("Audiovisual Projection Mapper: Grid Segmenter")
st.write(
    "Upload a source image and a binary grid mask. The preview will render the exact artwork mapped onto each seat shape."
)

col1, col2 = st.columns(2)
with col1:
    source_file = st.file_uploader(
        "Upload Source Image (Visual Content)", type=["jpg", "png", "jpeg"]
    )
with col2:
    grid_file = st.file_uploader(
        "Upload Grid Mask (White seats on Black background)",
        type=["jpg", "png", "jpeg"],
    )

if source_file and grid_file:
    # Read images into OpenCV format
    src_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
    source_img = cv2.imdecode(src_bytes, cv2.IMREAD_COLOR)

    grid_bytes = np.asarray(bytearray(grid_file.read()), dtype=np.uint8)
    grid_img = cv2.imdecode(grid_bytes, cv2.IMREAD_GRAYSCALE)

    # Match dimensions
    h, w = grid_img.shape[:2]
    if source_img.shape[:2] != (h, w):
        source_img = cv2.resize(
            source_img, (w, h), interpolation=cv2.INTER_CUBIC
        )

    # Threshold grid to create binary mask (255 = seat area, 0 = background)
    _, binary_mask = cv2.threshold(grid_img, 127, 255, cv2.THRESH_BINARY)

    # Find individual seat contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    st.success(f"Detected {len(contours)} distinct seat shapes.")

    # 1. APPLY FULL MASK: Retain artwork ONLY inside the white seat regions
    mapped_canvas = cv2.bitwise_and(source_img, source_img, mask=binary_mask)

    # 2. Extract seat individual crops & build JSON map
    seat_crops = {}
    json_mapping = {
        "grid_dimensions": {"width": w, "height": h},
        "total_seats": len(contours),
        "seats": [],
    }

    # Canvas for text/outline overlays if user toggles them
    overlay_canvas = mapped_canvas.copy()

    for idx, cnt in enumerate(contours):
        seat_id = f"Seat_{idx+1}"
        x, y, bw, bh = [int(v) for v in cv2.boundingRect(cnt)]
        center_x = int(x + bw / 2)
        center_y = int(y + bh / 2)

        # Draw green seat outlines and ID numbers on the overlay version
        cv2.drawContours(overlay_canvas, [cnt], -1, (0, 255, 0), 1)
        cv2.putText(
            overlay_canvas,
            str(idx + 1),
            (center_x - 5, center_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )

        # Create individual crop for seat listing
        single_seat_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(single_seat_mask, [cnt], -1, 255, -1)
        isolated_source = cv2.bitwise_and(
            source_img, source_img, mask=single_seat_mask
        )
        cropped_seat = isolated_source[y : y + bh, x : x + bw]

        seat_crops[seat_id] = cv2.cvtColor(cropped_seat, cv2.COLOR_BGR2RGB)

        json_mapping["seats"].append(
            {
                "id": seat_id,
                "index": idx + 1,
                "center": {"x": center_x, "y": center_y},
                "bounding_box": {"x": x, "y": y, "width": bw, "height": bh},
            }
        )

    # UI Options for Rendering Preview
    st.subheader("Mapped Projection Output")

    show_labels = st.checkbox("Show Seat IDs and Outlines", value=True)

    # Convert to RGB for display
    display_img = cv2.cvtColor(
        overlay_canvas if show_labels else mapped_canvas, cv2.COLOR_BGR2RGB
    )
    st.image(
        display_img,
        caption="Full Stadium Render (Image segments applied only onto seat geometry)",
        use_container_width=True,
    )

    # Export Section
    st.subheader("JSON Coordinates Output")
    json_str = json.dumps(json_mapping, indent=4)

    st.download_button(
        label="📄 Download JSON Seat Coordinates",
        data=json_str,
        file_name="seat_mapping.json",
        mime="application/json",
    )

    # Individual seat extracts
    st.subheader("Extracted Seat Segments")
    cols = st.columns(6)
    for idx, (seat_name, img_rgb) in enumerate(seat_crops.items()):
        col = cols[idx % 6]
        col.image(img_rgb, caption=seat_name, width=100)
