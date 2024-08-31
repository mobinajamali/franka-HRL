import pygame

pygame.init()
pygame.joystick.init()

joystick_count = pygame.joystick.get_count()
print(f"Number of joysticks detected: {joystick_count}")

if joystick_count > 0:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        print(f"Joystick {i} initialized: {joystick.get_name()}")
else:
    print("No joystick detected.")
