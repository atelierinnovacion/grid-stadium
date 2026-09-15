import json
import math
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Projection Mapping Seat Divider")

st.title("Audiovisual Projection Mapper: Image Tile Segmenter")
st.write(
    "Upload a source image and a binary grid mask. The image will be sliced into \(N\) equal segments (where \(N\) = total seats) and mapped onto each seat shape."
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
    # Read source image and grid mask
    src_bytes = np.asarray(bytearray(source_file.read()), dtype=np.uint8)
    source_img = cv2.imdecode(src_bytes, cv2.IMREAD_COLOR)

    grid_bytes = np.asarray(bytearray(grid_file.read()), dtype=np.uint8)
    grid_img = cv2.imdecode(grid_bytes, cv2.IMREAD_GRAYSCALE)

    # Threshold grid to binary (255 = seat area, 0 = background)
    _, binary_mask = cv2.threshold(grid_img, 127, 255, cv2.THRESH_BINARY)

    # Find individual seat contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    num_seats = len(contours)

    if num_seats == 0:
        st.error("No valid seat contours found in the grid mask!")
    else:
        st.success(f"Detected **{num_seats}** distinct seats.")

        # --- STEP 1: Sort seats spatially (Top-to-Bottom, Left-to-Right) ---
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        # Pair contour and bounding box for sorting
        seat_data = list(zip(contours, bounding_boxes))
        # Sort primarily by Y (row-wise), secondarily by X
        seat_data.sort(key=lambda item: (item[1][1] // 30, item[1][0]))

        # --- STEP 2: Partition Source Image into N Segments ---
        src_h, src_w = source_img.shape[:2]

        # Calculate optimal matrix dimensions (cols x rows) for N tiles
        cols = math.ceil(math.sqrt(num_seats))
        rows = math.ceil(num_seats / cols)

        tile_w = src_w // cols
        tile_h = src_h // rows

        image_tiles = []
        for r in range(rows):
            for c in range(cols):
                if len(image_tiles) >= num_seats:
                    break
                x1 = c * tile_w
                y1 = r * tile_h
                x2 = src_w if c == cols - 1 else (c + 1) * tile_w
                y2 = src_h if r == rows - 1 else (r + 1) * tile_h

                tile = source_img[y1:y2, x1:x2]
                image_tiles.append(tile)

        # --- STEP 3: Render Mapped Canvas & Prepare Output ---
        grid_h, grid_w = grid_img.shape[:2]
        output_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        overlay_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        json_mapping = {
            "grid_dimensions": {"width": grid_w, "height": grid_h},
            "total_seats": num_seats,
            "grid_matrix": {"rows": rows, "cols": cols},
            "seats": [],
        }

        seat_crops = {}

        for idx, (cnt, (x, y, bw, bh)) in enumerate(seat_data):
            seat_id = f"Seat_{idx+1}"
            center_x = int(x + bw / 2)
            center_y = int(y + bh / 2)

            # Get corresponding image tile
            tile = image_tiles[idx]

            # Resize image tile to fit the seat's bounding box
            resized_tile = cv2.resize(
                tile, (bw, bh), interpolation=cv2.INTER_CUBIC
            )

            # Create a localized mask for the seat contour
            mask_crop = np.zeros((bh, bw), dtype=np.uint8)
            cnt_shifted = cnt - [x, y]  # Shift contour relative to bounding box
            cv2.drawContours(mask_crop, [cnt_shifted], -1, 255, -1)

            # Mask the resized tile so it only fits within the contour
            masked_tile = cv2.bitwise_and(
                resized_tile, resized_tile, mask=mask_crop
            )

            # Paste the masked tile onto the main canvas
            roi = output_canvas[y : y + bh, x : x + bw]
            output_canvas[y : y + bh, x : x + bw] = cv2.add(roi, masked_tile)

            # Store cropped seat preview
            seat_crops[seat_id] = cv2.cvtColor(masked_tile, cv2.COLOR_BGR2RGB)

            # Save mapping coordinates to JSON
            json_mapping["seats"].append(
                {
                    "id": seat_id,
                    "index": idx + 1,
                    "center": {"x": center_x, "y": center_y},
                    "bounding_box": {"x": x, "y": y, "width": bw, "height": bh},
                    "assigned_tile_index": idx + 1,
                }
            )

        # Copy mapped canvas for diagnostic overlay
        overlay_canvas = output_canvas.copy()
        for idx, (_, (x, y, bw, bh)) in enumerate(seat_data):
            center_x = int(x + bw / 2)
            center_y = int(y + bh / 2)
            cv2.rectangle(
                overlay_canvas, (x, y), (x + bw, y + bh), (0, 255, 0), 1
            )
            cv2.putText(
                overlay_canvas,
                str(idx + 1),
                (center_x - 5, center_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

        # --- STREAMLIT UI RENDER ---
        st.subheader("Mapped Output Preview")
        show_labels = st.checkbox("Show Seat Bounding Boxes and IDs", value=True)

        display_img = cv2.cvtColor(
            overlay_canvas if show_labels else output_canvas, cv2.COLOR_BGR2RGB
        )
        st.image(
            display_img,
            caption=f"Image partitioned into {num_seats} tiles ({rows}x{cols} grid) and mapped to seats",
            use_container_width=True,
        )

        st.subheader("JSON Coordinates & Tile Mapping Output")
        json_str = json.dumps(json_mapping, indent=4)
        st.download_button(
            label="📄 Download JSON Mapping",
            data=json_str,
            file_name="seat_tile_mapping.json",
            mime="application/json",
        )

        st.subheader("Extracted Seat Tile Extracts")
        cols_ui = st.columns(6)
        for idx, (seat_name, img_rgb) in enumerate(seat_crops.items()):
            col = cols_ui[idx % 6]
            col.image(img_rgb, caption=seat_name, width=100)
