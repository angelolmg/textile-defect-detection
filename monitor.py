import PySimpleGUI as sg
from PIL import Image, ImageTk


def main():
    # Set the default values for the settings
    patch_size = 64
    resize_size = 512
    detection_confidence = 0.5
    conveyor_speed = 60
    model_file = 'models/yolov8s-cls_tilda400_50ep/yolov8s-cls_tilda400_50ep.pt'
    images = [
        Image.open(f"./images/027.jpg").resize((750,128)),
        Image.open(f"./images/003.jpg").resize((750,128)),
        Image.open(f"./images/372.jpg").resize((750,128)),
    ]
    index=0

    # Define the layout for each camera monitor section
    camera_layout_cam0 = [
        [sg.Image(key='-IMAGE_CAM_0-', size=(750, 128), pad=12)],
    ]

    camera_layout_cam1 = [
        [sg.Image(key='-IMAGE_CAM_1-', size=(750, 128), pad=12)],
    ]

    camera_layout_cam2 = [
        [sg.Image(key='-IMAGE_CAM_2-', size=(750, 128), pad=12)],
    ]

    # Define the layout for each tab in the stats section
    roll_map_layout = [
        [sg.Text('Roll Map')],  # Adjust the width of the tab
    ]

    summary_layout = [
        [sg.Text('Summary')],
    ]

    defects_layout = [
        [sg.Text('Defects')],
    ]

    # Define the layout for the dynamic information display section
    info_font = ('Any-Bold', 11, 'bold')
    info_layout = [
        [sg.Text('DPS: ', font=info_font), sg.Text('0', key='-DPS_VALUE-', size=(5, 1)),
         sg.Text('Speed (m/min): ', font=info_font), sg.Text('60', key='-SPEED_VALUE-', size=(5, 1)),
         sg.Text('Position (m): ', font=info_font), sg.Text('0', key='-POS_VALUE-', size=(5, 1)),
         sg.Text('Defect Count: ', font=info_font), sg.Text('0', key='-DEFECT_COUNT_VALUE-', size=(5, 1))]
    ]

    # Combine all the layouts into one main layout
    layout = [
        [sg.Menu([['&File', ['&Load feed', 'Open session',
                 'Save session', '---', '&Settings', 'E&xit']]])],
        [sg.TabGroup([[sg.Tab('Cam_0', camera_layout_cam0)], [sg.Tab(
            'Cam_1', camera_layout_cam1)], [sg.Tab('Cam_2', camera_layout_cam2)]])],
        [sg.TabGroup([[sg.Tab('Roll Map', roll_map_layout)], [sg.Tab('Summary', summary_layout)], [
                     sg.Tab('Defects', defects_layout)]], expand_x=True,expand_y=True)],  # Adjust the width of the tabgroup
        [sg.Column(info_layout, expand_x=True, element_justification='c')]
    ]

    # Create the main window with size 800x600
    main_window = sg.Window('Fabric Monitor', layout, size=(800, 600))

    while True:
        event, values = main_window.read(timeout=1000)

        if event == sg.WIN_CLOSED:
            break
        elif event == 'Exit' and sg.popup_yes_no('Are you sure you want to exit?', title='Confirm Exit') == 'Yes':
            break
        elif event == 'Load feed':
            # Add functionality to load a file here
            file_path = sg.popup_get_file('Select a file to load')
            # Handle file loading logic
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

        main_window["-IMAGE_CAM_0-"].update(data = ImageTk.PhotoImage(images[index%len(images)]))
        main_window["-IMAGE_CAM_1-"].update(data = ImageTk.PhotoImage(images[(index+1)%len(images)]))
        main_window["-IMAGE_CAM_2-"].update(data = ImageTk.PhotoImage(images[(index+2)%len(images)]))
        # Update the speed value text after applying the settings
        main_window['-SPEED_VALUE-'].update(conveyor_speed)
        index = index + 1

    main_window.close()


if __name__ == '__main__':
    main()
