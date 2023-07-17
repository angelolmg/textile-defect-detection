# TODO: add clicable grid to select errors
# TODO: add button to break image in blocks and save defective patches and good one separatly

import PySimpleGUI as sg
import os
import PIL.Image as Image
import io

# Initialize variables
image_list = []
current_image_index = 0
max_width = 512
max_height = 512

# Define the layout
menu_layout = [
    [sg.Button("Load Images")],
    [sg.Button("<<", key="-PREV-"), sg.Button(">>", key="-NEXT-")],
    [sg.Text("", key="-COUNTER-", size=(10, 1), justification="center")]
]

screen_layout = [
    [
        sg.Image(key="-IMAGE-", size=(max_width, max_height), pad=(0, 10))
    ],
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
    return resized_image

# Event loop
while True:
    event, values = window.read()

    # Handle events
    if event == sg.WINDOW_CLOSED:
        break
    elif event == "Load Images":
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

                # Convert resized image to bytes
                image_bytes = io.BytesIO()
                resized_image.save(image_bytes, format='PNG')

                window["-IMAGE-"].update(data=image_bytes.getvalue())
                current_image_index = 0
            else:
                sg.popup_error("No valid image files selected.")

    elif event in ("-PREV-", "-NEXT-"):
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

        # Convert resized image to bytes
        image_bytes = io.BytesIO()
        resized_image.save(image_bytes, format='PNG')

        window["-IMAGE-"].update(data=image_bytes.getvalue())

    # Update the image counter
    window["-COUNTER-"].update(f"{current_image_index + 1}/{len(image_list)}")

# Close the window
window.close()
