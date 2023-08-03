import os
import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCROLL_SPEED = 5

# Colors
WHITE = (255, 255, 255)

# Load images from the "images" folder and resize them to fit the screen
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            img = pygame.image.load(os.path.join(folder, filename))
            img = pygame.transform.scale(img, (SCREEN_WIDTH, SCREEN_HEIGHT))
            images.append(img)
    return images

def main():
    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Image Scroller")

    # Load images
    image_folder = "images"
    images = load_images_from_folder(image_folder)

    # Calculate the total height of the carousel
    total_height = len(images) * SCREEN_HEIGHT

    # Initial position for each image in a dictionary, starting above the camera view
    image_positions = {i: (-(len(images) - i) * SCREEN_HEIGHT) + SCREEN_HEIGHT for i in range(len(images))}

    # Variable to track if scrolling is paused
    paused = False

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # Space bar to pause/resume scrolling
                    paused = not paused
                elif event.key == pygame.K_r:  # 'r' key to restart the carousel
                    for i in range(len(images)):
                        image_positions[i] = (-(len(images) - i) * SCREEN_HEIGHT) + SCREEN_HEIGHT
                    paused = False

        # Update the carousel only if not paused
        if not paused:
            # Clear the screen
            screen.fill(WHITE)

            # Blit images on the screen
            for i, image in enumerate(images):
                y_pos = image_positions[i]
                screen.blit(image, (0, y_pos))

                # Update the position for the next frame
                y_pos += SCROLL_SPEED
                if y_pos >= total_height:
                    y_pos = -(len(images) - 1) * SCREEN_HEIGHT

                image_positions[i] = y_pos

        # Update the display
        pygame.display.update()

        # Set the frame rate (adjust as needed)
        clock.tick(60)

if __name__ == "__main__":
    main()
