# Working on Windows 10 (WSL 1)
# Run export DISPLAY=:0 and Xming (https://sourceforge.net/projects/xming/)

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

# Get images from feed only every FRAME_SKIP frames
# SECONDS TO SKIP = VIDEO ORIGINAL HEIGHT / (VIDEO ORIGINAL FPS * VIDEO SPEED PER FRAME IN PIXELS)
# SECONDS TO SKIP = 600 / (60 * 5) = 2
# FRAMES TO SKIP = (VIDEO ORIGINAL FPS * SECONDS TO SKIP) - 1 = (60 * 2) - 1 = 119
FRAME_SKIP = 119

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
    # Turn frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normalize it to train size then save to disk for inference
    normalized_frame = cv2.resize(gray_frame, (768, 512))
    save_path = os.path.join('frames', f'cam0_{frame_count}.jpg')
    cv2.imwrite(save_path, normalized_frame)


def update_camera_image(main_window, video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    # Create the 'frames' folder if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            # Process and save frame then update the image in the main window
            process_and_save_frame(last_frame, frame_count)
            main_window['-IMAGE_CAM_0-'].update(
                data=ImageTk.PhotoImage(Image.fromarray(cv2.resize(last_frame, (780, 128)))))

            # Break the loop if the video ends
            break

        last_frame = frame
        frame_count += 1

        # Every FRAME_SKIP frames
        # Process and save frame then update the image in the main window
        if frame_count == 1 or frame_count % FRAME_SKIP == 0:
            process_and_save_frame(frame, frame_count)

        main_window['-IMAGE_CAM_0-'].update(data=ImageTk.PhotoImage(
            Image.fromarray(cv2.resize(frame, (780, 128)))))

    # Release the video capture object
    cap.release()


def process_image(file_name, model):
    file_path = os.path.join('frames', file_name)
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

    print(f'Number of patches: {len(images)}')

    # Defect inference
    results = model.predict(source=images, conf=0.25)
    print(results[0].probs)
    # Filter defects
    marked_images = []
    top1 = []
    marked_coordinates = []
    for i in range(len(images)):
        if results[i].probs.top1 != 0 and results[i].probs.top1conf > 0.99:
            top1.append(int(results[i].probs.top1))
            marked_images.append(images[i])
            marked_coordinates.append(image_coordinates[i])

    print(f'Number of patches with defect: {len(marked_images)}')
    print(f'Ratio defect/good: {len(marked_images)/len(images)*100}%')

    # Save defect images to dictionary
    classes = ['good', 'hole', 'objects', 'oil spot', 'thread error']
    csv_file = 'defects.csv'
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

    save_entries_to_csv(csv_file, new_entries)

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

    save_path = os.path.join('detections', file_name)
    cv2.imwrite(save_path, input_image)


def cleanup_frames_folder():
    model = YOLO(
        "models/multiclass/yolov8s-cls_tilda400_50ep/weights/best.pt")
    print(model.names)

    frames_folder = 'frames'

    # Create the 'frames' folder if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')

    # Create the 'detections' folder if it doesn't exist
    if not os.path.exists('detections'):
        os.makedirs('detections')

    while True:
        time.sleep(1)
        files = os.listdir(frames_folder)
        if len(files) > 0:
            print(f"{len(files)} files inside 'frames' folder")

            # Process one image from the frames folder
            oldest_file = min(files, key=lambda f: os.path.getctime(
                os.path.join(frames_folder, f)))
            print(f"Processing: {oldest_file}")
            process_image(oldest_file, model)

            os.remove(os.path.join(frames_folder, oldest_file))
            print(f"Deleted: {oldest_file}")
        else:
            print("No files inside 'frames' folder")


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


def create_defect_scatter_plot(defects_data_csv_path):
    # Check if the 'defects.csv' file exists
    if not os.path.exists(defects_data_csv_path):
        print(f"No {defects_data_csv_path} file found.")
        return -1

    # Read data from the CSV file using pandas
    df = pd.read_csv(defects_data_csv_path)

    x_positions = []
    y_positions = []
    defect_class_color = []

    classes = {'hole': 'red', 'objects': 'blue',
               'oil spot': 'green', 'thread error': 'brown'}

    for index, entry in df.iterrows():
        frame_pos = int(entry['frame_pos'])
        frame_class = entry['class']
        pos_x = int(entry['pos_x'])
        pos_y = int(entry['pos_y'])

        # Considering 512px = 15cm, 0.3 is the approximate ratio px/cm
        x_positions.append(0.03 * (frame_pos * 512 + pos_y))
        y_positions.append(pos_x * 0.03)
        defect_class_color.append(classes[frame_class])

    # Lets say a limit of 80
    limit = 80
    # If any position goes over limit it breaks it down into multiple lists
    x_positions = split_list_by_limit(x_positions, limit)
    y_positions = split_list_into_structure(
        y_positions, [len(sublist) for sublist in x_positions])
    defect_class_color = split_list_into_structure(
        defect_class_color, [len(sublist) for sublist in x_positions])

    plot_index = -1

    for info in zip(x_positions, y_positions, defect_class_color):
        plot_index += 1
        x, y, c = info
        plt.figure(figsize=(7.3, 3.2), dpi=100)
        scatter = plt.scatter(x, y, marker='o', color=c)

        # Create a custom legend
        legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                    markerfacecolor=class_color) for class_name, class_color in classes.items()]

        plt.legend(handles=legend_labels, loc='upper right',
                   bbox_to_anchor=(1.28, 1.0))

        plt.xlim(limit * plot_index - 5, limit * (plot_index + 1))
        plt.ylim(-2, 24)
        plt.xlabel('Vertical position (cm)')
        plt.ylabel('Horizontal position (cm)')
        plt.grid(True)
        plt.savefig(f'rollmap_plot_{plot_index}.png', bbox_inches='tight')
        plt.close()

    return plot_index


def update_summary_data(window, defect_summary_data):
    window['-DEFECT_SUMMARY_TABLE-'].update(values=[[key, value]
                                            for key, value in defect_summary_data.items()])


def sort_table(table, sort_column_index):
    try:
        df = pd.DataFrame(table)
        sorted_df = df.sort_values(by=sort_column_index)
        sorted_table = sorted_df.values.tolist()
    except Exception as e:
        sg.popup_error('Error in sort_table', 'Exception in sort_table', e)
        sorted_table = table
    return sorted_table

def resfresh_defect_table(defects_data_csv_path, fallback):
    # Check if the CSV file exists
    if os.path.exists(defects_data_csv_path):
        defects_data = pd.read_csv(defects_data_csv_path)
    else: 
        defects_data = fallback

    return defects_data
    

def update_rollmap_view(window, current_rollmap, total_rollmaps):
    window['-CANVAS_ROLL_MAP-'].update(data=ImageTk.PhotoImage(
                Image.open(f'rollmap_plot_{current_rollmap}.png')))
    window['-ROLLMAP_INDEX-'].update(
        f'{current_rollmap+1}/{total_rollmaps+1}')

def main():
    # Start a thread to clean up the frames folder
    cleanup_thread = threading.Thread(target=cleanup_frames_folder)
    cleanup_thread.daemon = True
    cleanup_thread.start()

    # Set the default values for the settings
    patch_size = 64
    resize_size = 512
    detection_confidence = 0.5
    conveyor_speed = 60
    model_file = 'models/yolov8s-cls_tilda400_50ep/yolov8s-cls_tilda400_50ep.pt'
    rollmap_image_index = 0
    defects_data_csv_path = 'defects.csv'
    empty_df = pd.DataFrame(columns=['frame_pos', 'frame_index', 'camera', 'class',
                                    'pos_x', 'pos_y', 'date', 'img_base64'])

    defect_summary_data = {
        'DPS': 0,
        'Speed (m/min)': 0,
        'Position (m)': 0,
        'Defect Count': 0
    }

    # Define the layout for each camera monitor section
    camera_layout_cam0 = [
        [sg.Image(key='-IMAGE_CAM_0-', size=(780, 128), pad=12)],
    ]

    # camera_layout_cam1 = [
    #     [sg.Image(key='-IMAGE_CAM_1-', size=(750, 128), pad=12)],
    # ]

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
                  expand_y=True)]
    ]

    # Check if the CSV file exists
    if not os.path.exists(defects_data_csv_path):
        # Create an empty DataFrame if the file doesn't exist
        defects_data = empty_df
    else:
        defects_data = pd.read_csv(defects_data_csv_path)

    defects_layout = [
        [sg.Table(values=defects_data.iloc[:, :-1].values.tolist(), headings=defects_data.columns.tolist()[:-1],
                  auto_size_columns=True,
                  display_row_numbers=False,
                  justification='left',
                  num_rows=min(20, len(defects_data)),
                  key='-DEFECTS_TABLE-',
                  enable_events=True,
                  expand_x=True,
                  expand_y=True,
                  enable_click_events=True)],
                  [sg.Button('Refresh', key='-REFRESH_DEFECT_DATA-')],
    ]

    # Define the layout for the dynamic information display section
    info_font = ('Any-Bold', 11, 'bold')
    info_layout = [
        [sg.Text('DPS: ', font=info_font), sg.Text('0', key='-DPS_VALUE-', size=(5, 1)),
         sg.Text('Speed (m/min): ', font=info_font), sg.Text('60',
                                                             key='-SPEED_VALUE-', size=(5, 1)),
         sg.Text('Position (m): ', font=info_font), sg.Text(
             '0', key='-POS_VALUE-', size=(5, 1)),
         sg.Text('Defect Count: ', font=info_font), sg.Text('0', key='-DEFECT_COUNT_VALUE-', size=(5, 1))]
    ]

    # Combine all the layouts into one main layout
    layout = [
        [sg.Menu([['&File', ['&Load feed', 'Open session',
                             'Save session', '&Reset session', '---', '&Settings', 'E&xit']]])],
        [sg.TabGroup([
            [sg.Tab('Cam_0', camera_layout_cam0)],
            # [sg.Tab('Cam_1', camera_layout_cam1)],
            # [sg.Tab('Cam_2', camera_layout_cam2)]
        ], expand_x=True)],
        [sg.TabGroup([[sg.Tab('Roll Map', roll_map_layout)], [sg.Tab('Summary', summary_layout)], [
                     sg.Tab('Defects', defects_layout)]], expand_x=True, expand_y=True)],  # Adjust the width of the tabgroup
        [sg.Column(info_layout, expand_x=True, element_justification='c')]
    ]

    # Create the main window with size 800x600
    main_window = sg.Window('Fabric Monitor', layout)

    while True:
        event, values = main_window.read(timeout=1000)

        if event == sg.WIN_CLOSED:
            break
        elif event == 'Exit' and sg.popup_yes_no('Are you sure you want to exit?', title='Confirm Exit') == 'Yes':
            break
        elif event == 'Load feed':
            # Add functionality to load a video file here
            video_file = sg.popup_get_file('Select a video file to load')
            if video_file:
                # Create a separate thread for updating the camera image with video frames
                camera_thread = threading.Thread(
                    target=update_camera_image, args=(main_window, video_file))
                camera_thread.daemon = True
                camera_thread.start()
        elif event == 'Reset session':
            if sg.popup_yes_no('Are you sure you want to reset the session?\nThis will delete "defects.csv", roll map graphs\nand detections folder contents.', title='Confirm Reset') == 'Yes':
                if os.path.exists(defects_data_csv_path):
                    os.remove(defects_data_csv_path)
                # Remove images starting with 'rollmap_plot_'
                for file in os.listdir('.'):
                    if file.startswith('rollmap_plot_') and file.endswith('.png'):
                        os.remove(file)
                if os.path.exists('detections'):
                    for file in os.listdir('detections'):
                        file_path = os.path.join('detections', file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                defects_data = resfresh_defect_table(defects_data_csv_path, empty_df)
                print("Session reset completed.")
            
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
                [sg.Text('Conveyor belt speed (m/min)'),
                 sg.InputText(default_text=conveyor_speed, key='-CONVEYOR_SPEED-')],
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
                    conveyor_speed = int(values['-CONVEYOR_SPEED-'])
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
            defects_data = resfresh_defect_table(defects_data_csv_path, empty_df)

        # TABLE CLICKED Event has value in format ('-TABLE=', '+CLICKED+', (row,col))
        elif isinstance(event, tuple) and event[0] == '-DEFECTS_TABLE-':
            # Header was clicked and wasn't the "row" column
            if event[2][0] == -1 and event[2][1] != -1:
                col_num_clicked = event[2][1]
                col_name = defects_data.columns[col_num_clicked]
                new_table = defects_data.sort_values(
                    by=col_name, ascending=True)
                defects_data = new_table

        # Update the speed value text after applying the settings
        main_window['-SPEED_VALUE-'].update(conveyor_speed)
        main_window['-DEFECTS_TABLE-'].update(
                    values=defects_data.iloc[:, :-1].values.tolist())

        # Update the canvas element with the loaded image
        last_index = create_defect_scatter_plot(defects_data_csv_path)
        if last_index >= 0:
            update_rollmap_view(main_window, rollmap_image_index, last_index)
            defect_summary_data['Speed (m/min)'] = 60
            update_summary_data(main_window, defect_summary_data)
        else:
            rollmap_image_index = 0
            main_window['-CANVAS_ROLL_MAP-'].update(data=None)
            main_window['-ROLLMAP_INDEX-'].update('0/0')
            

    main_window.close()


if __name__ == '__main__':
    main()
