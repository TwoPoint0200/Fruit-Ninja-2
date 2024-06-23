import cv2
import numpy as np
import time
import keyboard
from ultralytics import YOLO
import os
import torch
from game import Game
from pynput.mouse import Button, Controller
import pygetwindow as gw
import bettercam


mouse = Controller()
cached_app_position = []
    

# Create a new camera object to capture the screen
camera = bettercam.create(output_color="BGR")
# Capture the screen the Fruit Ninja screen
# The dimensions for the screen capture are cached to avoid recalculating them every frame
def capture_window_screenshot(window_title, new_size=(640, 640)):
    try:
        # Check if the cached dimensions are available
        # If they are not, calculate the dimensions and cache them

        # NOTE: The dimensions are hardcoded for the BlueStacks window
        # These dimensions are for a 1920x1080 monitor with blustacks window maximized (not fullscreen)
        # and the right sidebar in BlueStacks hidden
        if len(cached_app_position) < 1:
            window = gw.getWindowsWithTitle(window_title)[0]
            app_x = window.left + 73
            app_width = window.width - 146
            app_y = window.top + 33
            app_height = window.height - 34
            app_x_w = app_x + app_width
            app_y_h = app_y + app_height
            cached_app_position.append((app_x, app_y, app_width, app_height, app_x_w, app_y_h))
        # Get the cached dimensions
        else:
            app_x, app_y, app_width, app_height, app_x_w, app_y_h = cached_app_position[0]

        # Capture the screen
        frame = camera.grab(region=(app_x, app_y, app_x_w, app_y_h))
        return frame, app_x, app_y, app_width, app_height
    except IndexError:
        # If the window is not found, throw an exception
        raise Exception(f"No window with title '{window_title}' found.")

def precise_sleep(duration):
    # Sleep for a precise duration as time.sleep is not accurate at very small durations
    start_time = time.time()
    while time.time() < start_time + duration:
        pass

def move(to_position, delay=1):
    # Move the mouse to the given position and wait for the given duration
    mouse.position = to_position
    precise_sleep(delay)

cached_angle = {}
def mouse_circle(radius, step_duration=1):
    # Move the mouse in a circle with the center at the current position and the given radius
    start_position = mouse.position

    # Number of points to have in the circle
    steps = 25

    # Check if the angles for the given radius are cached
    angles = cached_angle.get(radius, None)

    # If the angles are not cached, calculate them
    if (angles is None):
        angles = np.linspace(0, 2 * np.pi, steps)
        cos = np.cos(angles) * radius
        sin = np.sin(angles) * radius
        cached_angle[radius] = (cos, sin)
    # If the angles are cached, get the cosine and sine values
    else:
        cos, sin = angles

    # Calculate the new positions for the mouse using the cosine and sine values
    x_positions = start_position[0] + cos
    y_positions = start_position[1] + sin

    # Move the mouse to the new positions
    for i in range(steps):
        new_x = x_positions[i]
        new_y = y_positions[i]
        mouse.position = (new_x, new_y)
        precise_sleep(step_duration)

def cut_fruit(path: np.ndarray, app_x, app_y, app_width, app_height):
    # Cut the fruit along the path
    if len(path) == 0:
        return

    # Convert the path to the screen coordinates
    path[:, 0] = (path[:, 0] * app_width + app_x).astype(int)
    path[:, 1] = (path[:, 1] * app_height + app_y).astype(int)

    # Get the first point from the path
    first_point = path[0]
    path = path[1:]

    # Move the mouse to the first point
    mouse.position = tuple(first_point)

    # Start slicing the fruit
    mouse.press(Button.left)

    # Move the mouse in a circle to slice the first fruit
    mouse_circle(25, 0.0000002)
    # mouse_circle(25, 0.0000002)

    # Move the mouse to the rest of the points in the path
    time_per_point = 0.0000002
    # time_per_point = 0.0000002
    for point in path:
        # Move the mouse to the point
        move(tuple(point), time_per_point)
    # Move the mouse in a circle to slice the last fruit
    mouse_circle(25, 0.0000002)
    mouse.release(Button.left)

def main():
    if __name__ == '__main__':
        # Get the coordinates of the BlueStacks window by name
        app_name = "BlueStacks App Player"

        # Create a new window to display screenshots of the BlueStacks window
        # cv2.namedWindow('Live Screen', cv2.WINDOW_NORMAL)

        # Select device to use
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Set the GPU to use (0, 1, 2, or 3)
            print("Using GPU")

        # Load the YOLO model
        project_dir = os.getcwd()
        model_path = os.path.join(project_dir, 'models/FN3-17/weights/best.engine')
        model = YOLO(model_path, task='detect')
        # Do a prediction to load the model into memory
        dummy_input = torch.zeros((1, 3, 640, 640))
        model(dummy_input, device=device, half=True, imgsz=(640, 640))

        # Create a VideoWriter object
        # output_path = 'output_vid.mp4'
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = None

        # Create a new Game object
        game = Game()
        image_size = (640, 640)
        time_at_last_frame = 0
        while True:
            # start = time.time()
            # Check if the 'q' key has been pressed
            if keyboard.is_pressed('q'):
                break

            # Capture the screenshot of the desired region
            screenshot, app_x, app_y, app_width, app_height = capture_window_screenshot(app_name, new_size=image_size)
            # If the screenshot is None it means there is no new image so skip the current iteration since the frame has not changed
            if screenshot is None:
                continue

            # If the VideoWriter object is not created, create it
            # if video_writer is None:
            #     video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (screenshot.shape[1], screenshot.shape[0]))
                
            # Predict the screenshot using the YOLO model
            predictions = model(screenshot, device=device, verbose=False, iou=0.3, conf=0.6, int8=True, imgsz=image_size)
            

            # Get prediction classes and bounding boxes
            boxes = predictions[0].boxes
            classes = boxes.cls
            bounding_boxes = boxes.xywhn

            # Get an image with the bounding boxes and classes
            frame = predictions[0].plot()

            # Update the game state
            game.update(classes, bounding_boxes)

            # Get the path between the fruits that the fruit ninja should take
            cut_path = np.array(game.get_fruit_path())
            cut_fruit(cut_path, app_x, app_y, app_width, app_height)

            # Display the image with the bounding boxes and classes
            # cv2.imshow('Live Screen', frame)
            # end = time.time()

            # Write the frame to the video at 30 FPS
            # if end - time_at_last_frame > 0.0333333:
            #     video_writer.write(frame)
            #     time_at_last_frame = end

            # Wait for a short duration before capturing the next screenshot
            if cv2.waitKey(1) == ord('q'):  # waitKey argument is in milliseconds
                break


        # Release the video writer and close the video file
        # if video_writer is not None:
        #     video_writer.release()

        # Close the window and release the camera
        # cv2.destroyAllWindows()
        camera.release()


# Create a profile
# profiler = cProfile.Profile()
# profiler.enable()

# Run your main function
main()

# Disable the profiler after your function ends
# profiler.disable()

# # # Create a Stats object to format and print the profiler's data
# stats = pstats.Stats(profiler)

# # Sort the statistics by the cumulative time spent in the function
# stats.sort_stats(pstats.SortKey.CUMULATIVE)

# # Print the statistics
# stats.print_stats(30)