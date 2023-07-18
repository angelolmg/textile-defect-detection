import PySimpleGUI as sg
import os
import io
import cv2
import PIL.Image as Image
import numpy as np

# Initialize variables
image_list = []
current_image_index = 0
max_width = 512
max_height = 512
grid_size = 8
grid_thickness = 2

# Define the layout
menu_layout = [
    [sg.Button("Load Images")],
    [sg.Button("<<", key="-PREV-"), sg.Button(">>", key="-NEXT-")],
    [sg.Text("", key="-COUNTER-", size=(10, 1), justification="center")]
]

filename_text = sg.Text("", key="-FILENAME-", size=(30, 1), justification="center")

screen_layout = [
    [
        sg.Graph((max_width, max_height), (0, max_height), (max_width, 0), enable_events=True, key='-GRAPH-')
    ],
    [
        filename_text
    ]
]

layout = [
    [
        sg.Column(menu_layout, size=(100, 600), element_justification='center'),
        sg.VSeparator(),
        sg.Column(screen_layout, element_justification='center', expand_x=True)
    ]
]

# Create the window with a larger default size
window = sg.Window("Image Viewer", layout, size=(1000, 600))

# Function to resize image
def resize_image(filename, max_width, max_height):
    image = Image.open(filename)
    resized_image = image.resize((max_width, max_height), Image.LANCZOS)
    return np.array(resized_image)

# Function to draw grid on the image
def draw_grid(image):
    height, width, _ = image.shape
    cell_width = width // grid_size
    cell_height = height // grid_size

    for i in range(1, grid_size):
        image = cv2.line(image, (i * cell_width, 0), (i * cell_width, height), (0, 0, 0), thickness=grid_thickness)
        image = cv2.line(image, (0, i * cell_height), (width, i * cell_height), (0, 0, 0), thickness=grid_thickness)

    return image

def draw_highlight(image, highlights, color):
    for (start_point, end_point) in highlights:
        x1, y1 = start_point
        x2, y2 = end_point
        x, y, w, h = x1, y1, abs(x2-x1), abs(y2-y1)

        sub_img = image[y+2:y+h-1, x+2:x+w-1]
        color_rect = np.ones(sub_img.shape, dtype=np.uint8) * color
        res = cv2.addWeighted(sub_img.astype(np.float32), 1.0, color_rect.astype(np.float32), 0.15, 1.0)

        # Putting the image back to its position
        image[y+2:y+h-1, x+2:x+w-1] = res.astype(np.uint8)

    return image



# Event loop
while True:
    event, values = window.read()

    # Handle events
    if event == sg.WINDOW_CLOSED:
        break
    elif event == "Load Images":
        image_highlights = []
        # Open a file dialog to select images
        filenames = sg.popup_get_file(
            "Select Images",
            file_types=(("Images", "*.jpg *.jpeg *.png"),),
            multiple_files=True
        ).split(";")

        if filenames:
            # Clear existing image list
            image_list.clear()

            # Load and append valid images to the list
            for filename in filenames:
                if os.path.isfile(filename):
                    image_list.append(filename)

            if image_list:
                # Display the first image
                resized_image = resize_image(image_list[0], max_width, max_height)
                grid_image = draw_grid(resized_image)

                # Convert image to bytes
                image_bytes = Image.fromarray(grid_image.astype(np.uint8))
                byte_io = io.BytesIO()
                image_bytes.save(byte_io, format='PNG')

                window['-GRAPH-'].draw_image(data=byte_io.getvalue(), location=(0, 0))
                current_image_index = 0

                # Update the filename text
                filename_text.update(os.path.basename(image_list[current_image_index]))
            else:
                sg.popup_error("No valid image files selected.")

    elif event in ("-PREV-", "-NEXT-"):
        image_highlights = []
        # Check if image_list is empty
        if not image_list:
            continue

        # Navigate through the images
        if event == "-PREV-":
            current_image_index -= 1
        elif event == "-NEXT-":
            current_image_index += 1

        # Wrap around if the index goes out of bounds
        current_image_index %= len(image_list)

        # Resize the image and update the displayed image
        resized_image = resize_image(image_list[current_image_index], max_width, max_height)
        grid_image = draw_grid(resized_image)

        # Convert image to bytes
        image_bytes = Image.fromarray(grid_image.astype(np.uint8))
        byte_io = io.BytesIO()
        image_bytes.save(byte_io, format='PNG')

        window['-GRAPH-'].erase()
        window['-GRAPH-'].draw_image(data=byte_io.getvalue(), location=(0, 0))

        # Update the image counter
        window["-COUNTER-"].update(f"{current_image_index + 1}/{len(image_list)}")

        # Update the filename text
        filename_text.update(os.path.basename(image_list[current_image_index]))

    elif event == '-GRAPH-':
        # Get the clicked cell coordinates based on the click position
        cell_width = max_width // grid_size
        cell_height = max_height // grid_size
        x, y = values['-GRAPH-']
        cell_x = x // cell_width
        cell_y = y // cell_height

        # Update the displayed image
        resized_image = resize_image(image_list[current_image_index], max_width, max_height)
        grid_image = draw_grid(resized_image)
        
        # Highlight the clicked cell by drawing a rectangle
        top_left = (cell_x * cell_width, cell_y * cell_height)
        bottom_right = ((cell_x + 1) * cell_width, (cell_y + 1) * cell_height)

        if (top_left, bottom_right) not in image_highlights:
            image_highlights.append((top_left, bottom_right))
        else: image_highlights.remove((top_left, bottom_right))

        highlighted_image = draw_highlight(grid_image, image_highlights, (0, 255, 0))
        
        # Convert image to bytes
        image_bytes = Image.fromarray(highlighted_image.astype(np.uint8))
        byte_io = io.BytesIO()
        image_bytes.save(byte_io, format='PNG')

        window['-GRAPH-'].erase()
        window['-GRAPH-'].draw_image(data=byte_io.getvalue(), location=(0, 0))
        

# Close the window
window.close()
