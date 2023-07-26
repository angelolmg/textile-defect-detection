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
    [sg.Text("", key="-COUNTER-", size=(10, 1), justification="center")],
    [sg.Button("Patch")]
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

# Function to create and display the class selection layout
def show_class_selection_layout():
    class_names = []
    class_input_layout = [
        [sg.Text("Welcome to the Image Patcher!", justification="center", font=("Helvetica", 20, "bold"))],
        [sg.Text("First off, add the classes. The first class on the list will be the positive class (the default when\nthere's nothing of interest) and the second and beyond are the artifact classes.\nAdd at least two classes to proceed, the positive and one artifact (like a 'good' and 'defective' class)", justification="center", font=("Helvetica", 10))],
        [sg.Text("", size=(1, 1))],  # Empty row for vertical spacing
        [sg.Text("Enter class names, one per line:", font=("Helvetica", 12), justification="center")],
        [sg.InputText("", key="-CLASS-", size=(25, 1), font=("Helvetica", 12))],
        [sg.Button("Add Class", key="-ADD-", size=(25, 1), font=("Helvetica", 11))],
        [sg.Text("", size=(1, 1))],  # Empty row for vertical spacing
        [sg.Text("Class list:", font=("Helvetica", 12))],
        [sg.Listbox(values=class_names, size=(30, 5), key="-CLASS_LIST-", enable_events=True)],
        [sg.Text("", size=(1, 2))],  # Empty row for vertical spacing
        [sg.Button("Start", key="-START-", font=("Helvetica", 12), size=(25, 1), button_color=("white", "green"))]
    ]

    class_window = sg.Window("Image Patcher - Class Selection", class_input_layout, finalize=True, margins=(20, 20), size=(650, 500), element_justification='c')

    while True:
        event, values = class_window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == "-ADD-":
            class_name = values["-CLASS-"].strip()
            if class_name:
                class_names.append(class_name)
                class_window["-CLASS-"].update("")
                class_window["-CLASS_LIST-"].update(class_names)

        elif event == "-CLASS_LIST-":
            if values["-CLASS_LIST-"]:
                selected_class = values["-CLASS_LIST-"][0]
                if sg.popup_yes_no(f"Do you want to remove '{selected_class}'?", title="Remove Class") == "Yes":
                    class_names.remove(selected_class)
                    class_window["-CLASS_LIST-"].update(class_names)

        elif event == "-START-":
            if len(class_names) < 2:
                sg.popup_error("Please add at least 2 classes to proceed.\nThe first class will be the positive class, and the rest will be the artifact classes.")
            else:
                class_window.close()
                break

    return class_names

# Call the function to show the class selection layout and get class names
class_names = show_class_selection_layout()

if len(class_names) >= 2:

    # Create the window with a larger default size
    window = sg.Window("Image Patcher - Classifier", layout, size=(1000, 600))

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
        for (cell_x, cell_y) in highlights:
            top_left = (cell_x * 64, cell_y * 64)
            bottom_right = ((cell_x + 1) * 64, (cell_y + 1) * 64)
            x1, y1 = top_left
            x2, y2 = bottom_right
            x, y, w, h = x1, y1, abs(x2-x1), abs(y2-y1)

            sub_img = image[y+2:y+h-1, x+2:x+w-1]
            color_rect = np.ones(sub_img.shape, dtype=np.uint8) * color
            res = cv2.addWeighted(sub_img.astype(np.float32), 0.8, color_rect.astype(np.float32), 0.8, 1.0)

            # Putting the image back to its position
            image[y+2:y+h-1, x+2:x+w-1] = res.astype(np.uint8)

        return image

    # Function to create patches and save them
    def save_patch(image, name, patch_size, cell_x, cell_y):
        top_left = (cell_x * 64, cell_y * 64)
        bottom_right = ((cell_x + 1) * 64, (cell_y + 1) * 64)
        x1, y1 = top_left
        x2, y2 = bottom_right
        x, y, w, h = x1, y1, abs(x2-x1), abs(y2-y1)

        patch = image[y:y+h, x:x+w]
        patch_name = f"{name}_patch{cell_x}-{cell_y}.png"

        if (cell_x, cell_y) in image_highlights:
            patch_path = os.path.join("defective", patch_name)
        else:
            patch_path = os.path.join("good", patch_name)

        cv2.imwrite(patch_path, patch)

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

        elif event == "Patch":
            # Get the current working directory
            current_dir = os.getcwd()

            # Create "defective" and "good" folders
            defective_folder = os.path.join(current_dir, "defective")
            good_folder = os.path.join(current_dir, "good")
            os.makedirs(defective_folder, exist_ok=True)
            os.makedirs(good_folder, exist_ok=True)

            # Get the current image
            current_image_path = image_list[current_image_index]
            current_image = cv2.imread(current_image_path)
            image_name = os.path.basename(current_image_path).split(".")[0]

            # Resize the image to 512x512
            resized_image = cv2.resize(current_image, (512, 512))

            # Create patches
            patch_size = 64
            for cell_x in range(grid_size):
                for cell_y in range(grid_size):
                    save_patch(resized_image, image_name, patch_size, cell_x, cell_y)

            sg.popup("Patching completed!", title="Patch")

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
            if (cell_x, cell_y) not in image_highlights:
                image_highlights.append((cell_x, cell_y))
            else:
                image_highlights.remove((cell_x, cell_y))

            highlighted_image = draw_highlight(grid_image, image_highlights, (0, 255, 0))

            # Convert image to bytes
            image_bytes = Image.fromarray(highlighted_image.astype(np.uint8))
            byte_io = io.BytesIO()
            image_bytes.save(byte_io, format='PNG')

            window['-GRAPH-'].erase()
            window['-GRAPH-'].draw_image(data=byte_io.getvalue(), location=(0, 0))

        # Update the image counter
        window["-COUNTER-"].update(f"{current_image_index + 1}/{len(image_list)}")

    # Close the window
    window.close()
