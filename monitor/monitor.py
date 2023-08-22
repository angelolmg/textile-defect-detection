# Working on Windows 10 (WSL 1)
# Run Xming (https://sourceforge.net/projects/xming/) then export DISPLAY=:0

import PySimpleGUI as sg
from PIL import Image, ImageTk
import cv2
import threading
import os
import time
import ultralytics
from ultralytics import YOLO

import base64
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import io
import shutil

# Get images from feed only every FRAME_SKIP frames
# SECONDS TO SKIP = VIDEO ORIGINAL HEIGHT / (VIDEO ORIGINAL FPS * VIDEO SPEED PER FRAME IN PIXELS)
# SECONDS TO SKIP = 600 / (60 * 5) = 2
# FRAMES TO SKIP = (VIDEO ORIGINAL FPS * SECONDS TO SKIP) - 1 = (60 * 2) - 1 = 119
FRAME_SKIP = 119
CLOCK_SECS = 1
ROLLMAP_XLIMIT = 80

# Considering 512px = 15cm, 0.3 is the approximate ratio px/cm
CAM_FRAME_HEIGHT_PX = 512
CAM_FRAME_HEIGHT_CM = 15

# ANSI escape codes for text formatting
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# ANSI escape codes for text colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

def load_config_file(file_path):
    try:
        with open(file_path, "r") as config_file:
            config = json.load(config_file)

        # Update settings with values from the config file
        global patch_size, resize_size, detection_confidence, model_file, \
            defects_data_csv_path, session_date, detections_folder, frames_folder, rollmaps_folder

        patch_size = config.get("settings", {}).get("patch_size", patch_size)
        resize_size = config.get("settings", {}).get(
            "resize_size", resize_size)
        detection_confidence = config.get("settings", {}).get(
            "detection_confidence", detection_confidence)
        model_file = config.get("settings", {}).get("model_file", model_file)
        defects_data_csv_path = config.get("settings", {}).get(
            "defects_data_csv_path", defects_data_csv_path)
        session_date = config.get("session_date", session_date)
        detections_folder = config.get("detections_folder", detections_folder)
        frames_folder = config.get("frames_folder", frames_folder)
        rollmaps_folder = config.get("rollmaps_folder", rollmaps_folder)

        return True

    except Exception as e:
        sg.popup_error(f"Error loading config file: {str(e)}")
        return False

# Function to create the session folder and its subdirectories


def create_session_folders():
    # Get the current date and time
    current_time = time.strftime("%Y%m%d%H%M%S")
    session_folder = os.path.join("monitor/sessions", current_time)

    # Create the session folder
    os.makedirs(session_folder)

    # Create the 'detections' folder
    os.makedirs(os.path.join(session_folder, "detections"))

    # Create the 'frames' folder
    os.makedirs(os.path.join(session_folder, "frames"))

    # Create the 'roll maps' folder
    os.makedirs(os.path.join(session_folder, "rollmaps"))

    # Create a configuration file (config.json)
    config = {
        "session_date": current_time,
        "detections_folder": os.path.join(session_folder, "detections"),
        "frames_folder": os.path.join(session_folder, "frames"),
        "rollmaps_folder": os.path.join(session_folder, "rollmaps"),
        "settings": {
            "patch_size": 64,
            "resize_size": 512,
            "detection_confidence": 0.5,
            "model_file": "monitor/models/multiclass/yolov8s-cls_tilda400_50ep/weights/best.pt",
            "defects_data_csv_path": session_folder + "/defects.csv"
        }
    }

    with open(os.path.join(session_folder, "config.json"), "w") as config_file:
        json.dump(config, config_file)

    return session_folder


def save_entries_to_csv(csv_file, entries):
    # Check if CSV file already exists
    file_exists = os.path.exists(csv_file)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(entries)

    # If the file exists, append to it; otherwise, create a new file
    if file_exists:
        mode = 'a'
    else:
        mode = 'w'

    # Save the DataFrame to the CSV file
    df.to_csv(csv_file, mode=mode, index=False, header=not file_exists)


def read_entries_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    entries = df.values.tolist()
    return entries


def process_and_save_frame(frame, frame_count):
    global defect_summary_data
    # Turn frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize it to train size then save to disk for inference
    normalized_frame = cv2.resize(gray_frame, (768, 512))
    save_path = os.path.join(frames_folder, f'cam0_{frame_count}.jpg')
    cv2.imwrite(save_path, normalized_frame)
    defect_summary_data['Captures'] += 1


def update_camera_image(main_window, video_file):
    global start_session_time
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    start_session_time = time.time()

    # Create the 'frames' folder if it doesn't exist
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            if not paused: 
             # Process and save last frame then update the image in the main window
                process_and_save_frame(last_frame, frame_count)
                main_window['-IMAGE_CAM_0-'].update(
                    data=ImageTk.PhotoImage(Image.fromarray(cv2.resize(last_frame, (890, 128)))))

            else: print(RED + "[update_camera_image]" + RESET + " Currently paused. Resume to start collecting data.")
           
            # Break the loop if the video ends
            break

        last_frame = frame
        frame_count += 1

        # Every FRAME_SKIP frames
        # Process and save frame then update the image in the main window
        if frame_count == 1 or frame_count % FRAME_SKIP == 0:
            if not paused: process_and_save_frame(frame, frame_count)
            else: print(RED + "[update_camera_image]" + RESET + " Currently paused. Resume to start collecting data.")


        main_window['-IMAGE_CAM_0-'].update(data=ImageTk.PhotoImage(
            Image.fromarray(cv2.resize(frame, (890, 128)))))

    # Release the video capture object
    cap.release()


def process_image(file_name, model):
    global defect_summary_data

    file_path = os.path.join(frames_folder, file_name)
    input_image = cv2.imread(file_path)
    cell_size = 64
    rows, cols, _ = input_image.shape
    images = []
    image_coordinates = []

    # Cutout fabric patches
    for y in range(0, rows, cell_size):
        for x in range(0, cols, cell_size):
            image = input_image[y:y+cell_size, x:x+cell_size]
            images.append(image)
            image_coordinates.append((x, y, x+cell_size, y+cell_size))

    print(GREEN + "[process_image]" + RESET + f' Number of patches: {len(images)}')

    # Defect inference
    results = model.predict(source=images, conf=0.25)

    # Filter defects
    marked_images = []
    top1 = []
    marked_coordinates = []
    for i in range(len(images)):
        if results[i].probs.top1 != 0 and results[i].probs.top1conf > 0.99:
            top1.append(int(results[i].probs.top1))
            marked_images.append(images[i])
            marked_coordinates.append(image_coordinates[i])

    print(GREEN +"[process_image]" + RESET + f' Number of patches with defect: {len(marked_images)}')
    print(GREEN +"[process_image]" + RESET + f' Ratio defect/good: {len(marked_images)/len(images)*100}%')
    defect_summary_data['Defect Count'] += len(marked_images)

    # Save defect images to dictionary
    classes = ['good', 'hole', 'objects', 'oil spot', 'thread error']
    new_entries = []

    for i in range(len(marked_images)):

        # Convert the image to a base64 string
        _, buffer = cv2.imencode('.jpg', marked_images[i])
        base64_image = base64.b64encode(buffer).decode('utf-8')
        index = int(file_name.split('_')[1].split('.')[0])
        new_entries.append({'frame_pos': int(index/FRAME_SKIP),
                            'frame_index': index,
                            'camera': 'Cam_0',
                            'class': classes[top1[i]],
                            'pos_x': marked_coordinates[i][0],
                            'pos_y': marked_coordinates[i][1],
                            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'img_base64': base64_image})

    save_entries_to_csv(defects_data_csv_path, new_entries)

    color_mapping = {
        'good': (0, 255, 255),
        'hole': (0, 0, 255),
        'objects': (255, 0, 0),
        'oil spot': (0, 255, 0),
        'thread error': (203, 192, 255)
    }

    for i in range(len(marked_coordinates)):
        x1, y1, x2, y2 = marked_coordinates[i]
        cv2.rectangle(input_image, (x1, y1), (x2, y2),
                      color_mapping[new_entries[i]['class']], 2)

    save_path = os.path.join(
        'monitor/sessions', session_date, 'detections', file_name)
    cv2.imwrite(save_path, input_image)


def cleanup_frames_folder():
    while True:

        while True:
            time.sleep(CLOCK_SECS)
            if not os.path.exists(frames_folder):
                print(
                    BLUE + "[cleanup_frames_folder]" + RESET + f" '{frames_folder}' folder does not exist.")
                continue

            if not os.path.exists(model_file):
                print(
                    BLUE + "[cleanup_frames_folder]" + RESET + f" Model file '{model_file}' does not exist.")
                continue
            break

        model = YOLO(model_file)
        print(model.names)

        while True:
            time.sleep(CLOCK_SECS)
            try:
                files = os.listdir(frames_folder)
                if len(files) > 0:
                    print(BLUE + "[cleanup_frames_folder]" + RESET + f" {len(files)} files inside '{frames_folder}' folder")

                    # Process one image from the frames folder
                    oldest_file = min(files, key=lambda f: os.path.getctime(
                        os.path.join(frames_folder, f)))
                    print(f"[cleanup_frames_folder] Processing: {oldest_file}")
                    process_image(oldest_file, model)

                    os.remove(os.path.join(frames_folder, oldest_file))
                    print(BLUE + "[cleanup_frames_folder]" + RESET +  f" Deleted: {oldest_file}")
                else:
                    print(BLUE + "[cleanup_frames_folder]" + RESET +  f" No files inside '{frames_folder}' folder")

            except Exception as e:
                print(BLUE + "[cleanup_frames_folder]" + RESET +  f" Error while processing frames: {str(e)}")
                break


def split_list_by_limit(input_list, limit):
    result = []
    current_sum = 0
    curr_limit = limit
    sublist = []

    for value in input_list:
        if value <= curr_limit:
            sublist.append(value)
        else:
            result.append(sublist)
            sublist = [value]
            curr_limit += limit

    if sublist:
        result.append(sublist)

    return result


def split_list_into_structure(input_list, structure):
    result = []
    idx = 0

    for length in structure:
        sublist = input_list[idx: idx + length]
        result.append(sublist)
        idx += length

    return result


def create_defect_scatter_plot():
    global defect_summary_data
    # Check if the 'defects.csv' file exists
    if not os.path.exists(defects_data_csv_path):
        print(
            YELLOW + "[create_defect_scatter_plot]" + RESET + f" No {defects_data_csv_path} file found.")
        return -1

    # Read data from the CSV file using pandas
    df = pd.read_csv(defects_data_csv_path)

    x_positions = []
    y_positions = []
    defect_class_color = []

    # Get number of last detected frame to calculate actual cam position
    defect_summary_data['Position (m)'] = (
        (df.iloc[-1]['frame_pos'] + 1) + 1) * CAM_FRAME_HEIGHT_CM / 100

    classes = {'hole': 'red', 'objects': 'blue',
               'oil spot': 'green', 'thread error': 'brown'}

    for index, entry in df.iterrows():
        frame_pos = int(entry['frame_pos'])
        frame_class = entry['class']
        pos_x = int(entry['pos_x'])
        pos_y = int(entry['pos_y'])

        # For 512px = 15cm, 0.3 is the approximate ratio px/cm
        ratio = CAM_FRAME_HEIGHT_CM/CAM_FRAME_HEIGHT_PX
        x_positions.append(ratio * (frame_pos * CAM_FRAME_HEIGHT_PX + pos_y))
        y_positions.append(pos_x * ratio)
        defect_class_color.append(classes[frame_class])

    # If any position goes over limit it breaks it down into multiple lists
    x_positions = split_list_by_limit(x_positions, ROLLMAP_XLIMIT)

    # Match the same broken down structure of x_position to the others
    y_positions = split_list_into_structure(
        y_positions, [len(sublist) for sublist in x_positions])
    defect_class_color = split_list_into_structure(
        defect_class_color, [len(sublist) for sublist in x_positions])

    plot_index = -1

    for info in zip(x_positions, y_positions, defect_class_color):
        plot_index += 1
        x, y, c = info
        plt.figure(figsize=(8.7, 3), dpi=100)
        scatter = plt.scatter(x, y, marker='o', color=c)

        # Create a custom legend
        legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                    markerfacecolor=class_color) for class_name, class_color in classes.items()]

        plt.legend(handles=legend_labels, loc='upper right',
                   bbox_to_anchor=(1.24, 1.0))

        plt.xlim(ROLLMAP_XLIMIT * plot_index - 5,
                 ROLLMAP_XLIMIT * (plot_index + 1))
        plt.ylim(-2, 24)
        plt.xlabel('Vertical position (cm)')
        plt.ylabel('Horizontal position (cm)')
        plt.grid(True)
        save_path = os.path.join(
            rollmaps_folder, f'rollmap_plot_{plot_index}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    return plot_index


def update_summary_data(window):
    global defect_summary_data
    defect_summary_data['Elapsed time (s)'] = round(
        end_session_time - start_session_time, 1)

    window['-DEFECT_SUMMARY_TABLE-'].update(values=[[key, value]
                                            for key, value in defect_summary_data.items()])
    window['-ELAPSED_TIME_VALUE-'].update(
        defect_summary_data['Elapsed time (s)'])
    window['-CAP_VALUE-'].update(defect_summary_data['Captures'])
    window['-SPEED_VALUE-'].update(defect_summary_data['Speed (m/min)'])
    window['-POS_VALUE-'].update(defect_summary_data['Position (m)'])
    window['-DEFECT_COUNT_VALUE-'].update(defect_summary_data['Defect Count'])


def sort_table(table, sort_column_index):
    try:
        df = pd.DataFrame(table)
        sorted_df = df.sort_values(by=sort_column_index)
        sorted_table = sorted_df.values.tolist()
    except Exception as e:
        sg.popup_error('Error in sort_table', 'Exception in sort_table', e)
        sorted_table = table
    return sorted_table


def resfresh_defect_table(defect_csv, fallback):
    # Check if the CSV file exists
    if os.path.exists(defect_csv):
        defects_data = pd.read_csv(defect_csv)
    else:
        defects_data = fallback
        print(f"Defects CSV '{defect_csv}' does not exist.")

    return defects_data


def update_rollmap_view(window, current_rollmap, total_rollmaps):
    rollmap_path = os.path.join(
        rollmaps_folder, f'rollmap_plot_{current_rollmap}.png')
    window['-CANVAS_ROLL_MAP-'].update(
        data=ImageTk.PhotoImage(Image.open(rollmap_path)))
    window['-ROLLMAP_INDEX-'].update(f'{current_rollmap+1}/{total_rollmaps+1}')


def update_popup_with_row_info(row_data):
    # Decode the base64-encoded image data to bytes
    img_bytes = base64.b64decode(row_data["img_base64"])

    # Create a PIL image from the bytes
    img_pil = Image.open(io.BytesIO(img_bytes))

    # Resize the PIL image to 256x256
    img_pil_resized = img_pil.resize((128, 128))

    # Save the resized PIL image to a file
    save_path = os.path.join("monitor/sessions", session_date, "defect.png")
    img_pil_resized.save(save_path)

    popup_layout = [
        [sg.Text('Defect Details', font=('Any', 14, 'bold'))],
        [sg.Text(f'Frame Position: {row_data["frame_pos"]}')],
        [sg.Text(f'Frame Index: {row_data["frame_index"]}')],
        [sg.Text(f'Camera: {row_data["camera"]}')],
        [sg.Text(f'Class: {row_data["class"]}')],
        [sg.Text(
            f'Position (X, Y): ({row_data["pos_x"]}, {row_data["pos_y"]})')],
        [sg.Text(f'Date: {row_data["date"]}')],
        [sg.Image(filename=save_path, key='-DEFECT_IMAGE-', size=(400, 400))],
        [sg.Button('Close')]
    ]

    popup_window = sg.Window('Defect Details', popup_layout, modal=True)

    while True:
        event, values = popup_window.read()
        if event in (sg.WINDOW_CLOSED, 'Close'):
            break

    popup_window.close()


def load_default_settings():
    global patch_size, resize_size, detection_confidence, model_file, \
        defects_data_csv_path, session_date, detections_folder, frames_folder, rollmaps_folder, \
        paused, start_session_time, end_session_time, defect_summary_data

    # Settings
    patch_size = 64
    resize_size = 512
    detection_confidence = 0.5
    model_file = ""
    defects_data_csv_path = ""
    session_date = ""
    detections_folder = ""
    frames_folder = ""
    rollmaps_folder = ""

    # General info about session
    paused = False
    start_session_time = 0
    end_session_time = 0
    defect_summary_data = {
        'Elapsed time (s)': 0,
        'Captures': 0,
        'Speed (m/min)': 49.6,
        'Position (m)': 0,
        'Defect Count': 0
    }


def main():
    sg.theme('DarkTeal9')

    global patch_size, resize_size, detection_confidence, model_file, \
        defects_data_csv_path, session_date, detections_folder, frames_folder, rollmaps_folder, \
        paused, start_session_time, end_session_time, defect_summary_data

    # Start a thread to clean up the frames folder
    cleanup_thread = threading.Thread(target=cleanup_frames_folder)
    cleanup_thread.daemon = True
    cleanup_thread.start()

    # Set the default values for the settings
    load_default_settings()

    rollmap_image_index = 0
    empty_df = pd.DataFrame(columns=['frame_pos', 'frame_index', 'camera', 'class',
                                     'pos_x', 'pos_y', 'date', 'img_base64'])

    # Define the layout for each camera monitor section
    camera_layout_cam0 = [
        [sg.Image(key='-IMAGE_CAM_0-', size=(780, 128), pad=12)],
    ]

    camera_layout_cam1 = [
        [sg.Image(key='-IMAGE_CAM_1-', size=(750, 128), pad=12)],
    ]

    # camera_layout_cam2 = [
    #     [sg.Image(key='-IMAGE_CAM_2-', size=(750, 128), pad=12)],
    # ]

    # Define the layout for each tab in the stats section
    roll_map_layout = [
        [sg.Image(key='-CANVAS_ROLL_MAP-',
                  size=(780, 300), pad=10, expand_y=True)],
        [sg.Column([[sg.Button('<<', key='previous'),
                     sg.Text('Roll Map:', pad=((20, 0), 0)),
                     sg.Text('0/0', key='-ROLLMAP_INDEX-', pad=((0, 20), 0)),
                     sg.Button('>>', key='next')]], justification='center')]
    ]

    summary_layout = [
        [sg.Table(values=[[key, value] for key, value in defect_summary_data.items()],
                  headings=['Key', 'Value'],
                  auto_size_columns=False,
                  display_row_numbers=False,
                  num_rows=len(defect_summary_data)+20,
                  col_widths=[20, 10],
                  key='-DEFECT_SUMMARY_TABLE-',
                  justification='left',
                  expand_y=True, expand_x=True)]
    ]

    # Check if the CSV file exists
    if not os.path.exists(defects_data_csv_path):
        # Assign an empty DataFrame if the file doesn't exist
        defects_data = empty_df
    else:
        defects_data = pd.read_csv(defects_data_csv_path)

    defects_layout = [
        [sg.Table(values=defects_data.iloc[:, :-1].values.tolist(),
                  headings=defects_data.columns.tolist()[:-1],
                  auto_size_columns=False,
                  display_row_numbers=False,
                  justification='left',
                  num_rows=min(20, len(defects_data)),
                  col_widths=[10, 10, 10, 10, 10, 10, 10],
                  key='-DEFECTS_TABLE-',
                  enable_events=True,
                  expand_x=True,
                  expand_y=True,
                  enable_click_events=True)],
        [sg.Button('Refresh', key='-REFRESH_DEFECT_DATA-')],
    ]

    # Define the layout for the dynamic information display section
    info_font = ('Any-Bold', 11, 'bold')
    info_frame = [
        [sg.Button('▶', key='-START_BUTTON-', size=(4, 2)), sg.Button('❚❚', key='-PAUSE_BUTTON-', size=(4, 2)),
         sg.Text('Elapsed time (s): ', font=info_font), sg.Text(
             '0', key='-ELAPSED_TIME_VALUE-', size=(5, 1)),
         sg.Text('Captures: ', font=info_font), sg.Text(
             '0', key='-CAP_VALUE-', size=(5, 1)),
         sg.Text('Speed (m/min): ', font=info_font), sg.Text('0',
                                                             key='-SPEED_VALUE-', size=(5, 1)),
         sg.Text('Position (m): ', font=info_font), sg.Text(
            '0', key='-POS_VALUE-', size=(5, 1)),
         sg.Text('Defect Count: ', font=info_font), sg.Text('0', key='-DEFECT_COUNT_VALUE-', size=(5, 1))]
    ]

    info_layout = [
        [sg.Frame('', info_frame, expand_y=True)]
    ]

    # Combine all the layouts into one main layout
    layout = [
        [sg.Menu([['&File', ['&New session', '&Open session',
                             '&Delete session', '---', '&Settings', 'E&xit']]])],
        [sg.TabGroup([
            [sg.Tab('Cam_0', camera_layout_cam0)],
            [sg.Tab('Cam_1', camera_layout_cam1)],
            # [sg.Tab('Cam_2', camera_layout_cam2)]
        ], expand_x=True)],
        [sg.TabGroup([[sg.Tab('Roll Map', roll_map_layout)], [sg.Tab('Summary', summary_layout)], [
                     sg.Tab('Defects', defects_layout)]], expand_x=True, expand_y=True)],
        [sg.Column(info_layout, expand_x=True, element_justification='c')]
    ]

    # Create the main window with size 800x600
    main_window = sg.Window('Fabric Monitor', layout,
                            enable_close_attempted_event=True, resizable=False)
    while True:
        event, values = main_window.read(timeout=CLOCK_SECS*1000)

        if (event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT or event == 'Exit') and \
                sg.popup_yes_no('Are you sure you want to exit?', title='Confirm Exit') == 'Yes':
            break
        elif event == 'New session':
            # Add functionality to load a video file here
            video_file = sg.popup_get_file('Select a video file to load', file_types=(("MP4 Files", "*.mp4"), ("MOV Files", "*.mov")))
            if video_file:
                if video_file.lower().endswith((".mp4", ".mov")):
                    # Create a new session folder with subdirectories
                    session_folder = create_session_folders()
                    load_config_file(os.path.join(session_folder, "config.json"))
                    update_summary_data(main_window)

                    # Create a separate thread for updating the camera image with video frames
                    camera_thread = threading.Thread(
                        target=update_camera_image, args=(main_window, video_file))
                    camera_thread.daemon = True
                    camera_thread.start()
                else:
                    sg.popup_error("Invalid File", "Please select a .mp4 or .mov video file.")

        elif event == 'Open session':
            config_file_path = sg.popup_get_file(
                'Select a configuration JSON file to open', initial_folder="monitor/sessions")
            if config_file_path:
                if load_config_file(config_file_path):
                    sg.popup(
                        f"Configuration loaded successfully from {config_file_path}")

        elif event == 'Delete session':
            if sg.popup_yes_no('Are you sure you want to delete this session?\nAll contents in session folder will be lost.', title='Confirm Delete') == 'Yes':
                if session_date:
                    session_folder = os.path.join(
                        "monitor/sessions", session_date)

                    # Check if the session folder exists
                    if os.path.exists(session_folder):
                        # Delete everything inside the session folder
                        shutil.rmtree(session_folder)

                        print("Session contents deleted.")

                    # Set default settings values
                    load_default_settings()
                    update_summary_data(main_window)

                    # Clear the defects data
                    defects_data = empty_df
                    print("Session deletion completed.")
                else:
                    print("No session loaded, so nothing to delete.")

        elif event == 'Settings':
            # Open the settings window when "Settings" is clicked
            settings_layout = [
                [sg.Text('Settings')],
                [sg.Text('Patch size'), sg.Slider(range=(
                    16, 128), default_value=patch_size, resolution=16, orientation='h', key='-PATCH_SIZE-')],
                [sg.Text('Resize size'), sg.Slider(range=(
                    256, 1024), default_value=resize_size, resolution=64, orientation='h', key='-RESIZE_SIZE-')],
                [sg.Text('Detection confidence'), sg.Slider(range=(0, 1), default_value=detection_confidence,
                                                            resolution=0.01, orientation='h', key='-DETECTION_CONFIDENCE-')],
                [sg.Text('Model File'), sg.InputText(
                    default_text=model_file, key='-MODEL_FILE-'), sg.FileBrowse()],
                [sg.Button('Apply'), sg.Button('Cancel')]
            ]

            settings_window = sg.Window('Settings', settings_layout)

            while True:
                event, values = settings_window.read()

                if event == sg.WIN_CLOSED or event == 'Cancel':
                    break
                elif event == 'Apply':
                    # Get the updated settings values
                    patch_size = int(values['-PATCH_SIZE-'])
                    resize_size = int(values['-RESIZE_SIZE-'])
                    detection_confidence = float(
                        values['-DETECTION_CONFIDENCE-'])
                    model_file = values['-MODEL_FILE-']
                    # Apply the settings here (you can save them to a configuration file or use them in the main window)
                    settings_window.close()  # Close the settings window after applying the settings

            settings_window.close()  # Close the settings window if the "Cancel" button is clicked

        # Handle "next" and "previous" button clicks
        elif event == 'next':
            rollmap_image_index = min(rollmap_image_index + 1, last_index)
        elif event == 'previous':
            rollmap_image_index = max(rollmap_image_index - 1, 0)
        elif event == '-REFRESH_DEFECT_DATA-':
            defects_data = resfresh_defect_table(
                defects_data_csv_path, empty_df)
        elif event == '-START_BUTTON-':
            paused = False
        elif event == '-PAUSE_BUTTON-':
            paused = True 


        # TABLE CLICKED Event has value in format ('-TABLE=', '+CLICKED+', (row,col))
        elif isinstance(event, tuple) and event[0] == '-DEFECTS_TABLE-':
            # Header was clicked and wasn't the "row" column
            if event[2][0] == -1 and event[2][1] != -1:
                col_num_clicked = event[2][1]
                col_name = defects_data.columns[col_num_clicked]
                new_table = defects_data.sort_values(
                    by=col_name, ascending=True)
                defects_data = new_table
            # Clicked on a row and the row is not null
            elif event[1] == '+CLICKED+' and event[2][0] != None:
                clicked_row_index = event[2][0]
                clicked_row_data = defects_data.iloc[clicked_row_index].to_dict(
                )
                update_popup_with_row_info(clicked_row_data)

        # Update the speed value text after applying the settings
        main_window['-SPEED_VALUE-'].update(
            defect_summary_data['Speed (m/min)'])
        main_window['-DEFECTS_TABLE-'].update(
            values=defects_data.iloc[:, :-1].values.tolist())

        # Update the canvas element with the loaded image
        last_index = create_defect_scatter_plot()
        if last_index >= 0:
            end_session_time = time.time()
            update_rollmap_view(main_window, rollmap_image_index, last_index)
            update_summary_data(main_window)
        else:
            rollmap_image_index = 0
            main_window['-CANVAS_ROLL_MAP-'].update(data=None)
            main_window['-ROLLMAP_INDEX-'].update('0/0')

    main_window.close()


if __name__ == '__main__':
    main()
