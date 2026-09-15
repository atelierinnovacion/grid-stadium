import streamlit as st
import numpy as np
import cv2
import json
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Audiovisual Projection Mapper", layout="wide")

st.title("Projection & Stadium Seat Mapping Generator")
st.write("Upload a target visual image and define region coordinates to map specific image sectors to physical seats.")

col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader("1. Upload Target Image", type=["jpg", "png", "jpeg"])
with col2:
    grid_file = st.file_uploader("2. Upload Grid Coordinates (JSON)", type=["json"])

def generate_default_grid(width, height, rows=10, cols=10):
    """Generates a simple fallback grid if no file is provided."""
    grid = []
    dx = width / cols
    dy = height / rows
    for r in range(rows):
        for c in range(cols):
            grid.append({
                "id": f"Seat_{r+1}_{c+1}",
                "poly": [
                    [int(c * dx), int(r * dy)],
                    [int((c + 1) * dx), int(r * dy)],
                    [int((c + 1) * dx), int((r + 1) * dy)],
                    [int(c * dx), int((r + 1) * dy)]
                ]
            })
    return grid

if img_file is not None:
    # Load Image
    image = Image.open(img_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    # Parse or Generate Grid
    if grid_file is not None:
        grid_data = json.load(grid_file)
    else:
        st.info("No custom grid uploaded. Generating a sample 10x10 seating grid.")
        grid_data = generate_default_grid(w, h)

    mapped_results = []
    output_img = img_np.copy()

    # Process each region in the grid
    for item in grid_data:
        seat_id = item["id"]
        pts = np.array(item["poly"], dtype=np.int32)

        # Create mask for bounding polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Extract mean RGB color within polygon boundary
        mean_val = cv2.mean(img_np, mask=mask)
        rgb_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
        hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"

        # Draw extracted color back into preview image for verification
        cv2.fillPoly(output_img, [pts], rgb_color)
        cv2.polylines(output_img, [pts], True, (255, 255, 255), 1)

        mapped_results.append({
            "seat_id": seat_id,
            "r": rgb_color[0],
            "g": rgb_color[1],
            "b": rgb_color[2],
            "hex": hex_color
        })

    # Render results
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Source Visual")
        st.image(image, use_container_width=True)
    with res_col2:
        st.subheader("Mapped Projection Preview")
        st.image(output_img, use_container_width=True)

    # Data export options
    st.subheader("Mapped Output Data")
    df = pd.DataFrame(mapped_results)
    st.dataframe(df.head(10))

    json_str = json.dumps(mapped_results, indent=2)
    st.download_button(
        label="Download Mapping Configuration (JSON)",
        data=json_str,
        file_name="seat_mapping.json",
        mime="application/json"
    )