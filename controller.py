import numpy as np
import pygame

class FrankaController:
    def __init__(self):
        self.gripper_closed = None
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def get_action(self):
        action = np.zeros(9)
        gripper_button_pressed = False
        ## joystick axis
        # map left joystrick to joint 1 and joint 2 angular velocity
        action[0] = self.joystick.get_axis(0)  
        action[1] = self.joystick.get_axis(1)  
        # rearrange so that if pull down the joint goes down
        action[0] = action[0] * -1
        action[1] = action[1] * -1

        # map right joystrick to joint 3 and joint 4 
        action[2] = self.joystick.get_axis(3) 
        action[3] = self.joystick.get_axis(2)  
        action[3] = action[3] * -1

        ## joystick buttons
        if self.joystick.get_button(0):
            action[4] = -1
            print('button 0 pressed') 
        elif self.joystick.get_button(2):
            action[4] = 1
            print('button 2 pressed')
        elif self.joystick.get_button(1):
            self.gripper_closed = True
            gripper_button_pressed = True
            print('button 1 presses')
        elif self.joystick.get_button(3):
            self.gripper_closed = False
            gripper_button_pressed = True
            print('button 3 presses')
        elif self.joystick.get_button(4):
            action[5] = 1
            print('button 4 pressed')
        elif self.joystick.get_button(5):
            action[5] = -1
            print('button 5 pressed')
        elif self.joystick.get_button(6):
            action[6] = -1
            print('button 6 pressed')
        elif self.joystick.get_button(7):
            action[6] = 1
            print('button 7 pressed')
        elif self.joystick.get_button(8):
            action[7] = 1
            print('button 8 pressed')
        elif self.joystick.get_button(9):
            action[7] = -1
            print('button 9 pressed')
        
        mask = np.abs(action) >= 0.1
        action = action * mask
        # zero out action if it is not greater than 0.1
        action = np.where(action == -0.0, 0.0, action)
        if np.all(action == 0) and gripper_button_pressed == False:
            action = None
        else:
            if self.gripper_closed == True:
                action[7] = -1.0
                action[8] = -1.0
            elif self.gripper_closed == False:
                action[7] = 1.0
                action[8] = 1.0
        
        return action



