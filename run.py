import pygame, sys, os
from pygame.math import Vector2
import datetime
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog as fd

from cnn import *
import cnn

class Game(object):

    def Log(self, info):
        text = self.font.render(info, True, (126, 150, 189))
        self.screen.blit(text, (self.w-self.uiw + 5, self.h - 150))

    def __init__(self):
        #config
        self.Xres = 1000
        self.Yres = 600
        bgColor = (23, 23, 34)
        
        self.w = self.Xres
        self.h = self.Yres
        self.canvasMargin = 20
        self.uiw = 330
        self.appFolder = os.path.dirname(os.path.abspath(__file__))
        self.full_path = os.path.join(self.appFolder, "agency_fb.ttf")
        self.session = random.randint(1, 1000)
        
        self.debugMode = True

        self.logText = "None"

        #initialization
        pygame.init()
        self.screen = pygame.display.set_mode((self.Xres, self.Yres), pygame.RESIZABLE)

        r = pygame.Rect(0, 0, 0, 0)

        self.font = pygame.font.Font(self.full_path, 20)
        font3 = pygame.font.Font(self.full_path, 30)


        filename = ''
        tbactive = False

        self.texture = []
        self.textures = []

        self.history = []

        #menu
        self.menuWidth = 900
        self.menuX = self.Xres - self.menuWidth
        self.baseColor = (126, 150, 189)
        self.inactiveColor = (40, 54, 77)
        self.buttonPressed = True

        ############## SETUP #############
        self.texturSize = 28
        self.pixelBorderColor = [0, 0, 0]
        self.textureBorderColor = (117, 117, 117)
        pixelSize = 15
        self.textureSizeOnCanvas = self.texturSize*pixelSize

        #Dataset
        self.numberOfClasses = 26
        self.imageSize = 28
        self.supportVector = []
        self.classes = []

        #Prediction result
        self.numberOfBest = 10
        predBox = pygame.Rect(self.textureSizeOnCanvas + 40, 20, 150, 35*self.numberOfBest)

        

        ##################################

        ############# BUTTONS ############

        self.texts = ['LOAD DATASET', 'LOAD MODEL', 'TRAIN', 'CLEAR PAGE']
        self.buttonActive = [True, True, False, True]
        self.colors = [self.baseColor, self.baseColor, self.inactiveColor, self.baseColor]

        self.buttons = [   pygame.Rect(self.w-self.uiw+10, 5+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130, 5+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130+130, 5+30, 125, 25),

                                pygame.Rect(self.w-self.uiw+10, 5+30+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130, 5+30+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130+130, 5+30+30, 125, 25),

                                pygame.Rect(self.w-self.uiw+10, 5+60+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130, 5+60+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130+130, 5+60+30, 125, 25),

                                pygame.Rect(self.w-self.uiw+10, 5+90+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130, 5+90+30, 125, 25),
                                # pygame.Rect(self.menuX+5+130+130, 5+90+30, 125, 25),

                                ]

        ##################################

        ##################################

        #For CNN
        self.datasetDir = None
        self.trained = False
        self.modelLoaded = False

        self.predClasses = []
        self.pos = 0
        self.images = []

        ##################################


        self.textures.append(pygame.Rect(self.canvasMargin, self.canvasMargin, self.textureSizeOnCanvas, self.textureSizeOnCanvas))

        for i in range(self.texturSize):
            tmp = []
            for j in range(self.texturSize):
                tmp.append([pygame.Rect(i*pixelSize+self.canvasMargin, j*pixelSize+self.canvasMargin, pixelSize, pixelSize), [0, 0, 0], 1].copy())
            self.texture.append(tmp.copy())
            
        self.outputTexture = pygame.Rect(0, self.Yres - self.texturSize, self.texturSize, self.texturSize)

        while True:

            
            textBox = pygame.Rect(self.w-self.uiw+10, self.h-50-50, self.uiw-20, 40)
            saveButton = pygame.Rect(self.w-self.uiw+10, self.h-50, self.uiw-20, 40)

            self.w, self.h = pygame.display.get_surface().get_size()
            sub = self.screen.subsurface(self.outputTexture)
            self.mousePos = Vector2(pygame.mouse.get_pos())


            #rendering
            self.screen.fill(bgColor)

            uiRect = pygame.Rect(self.w-self.uiw, 0, self.uiw, self.h)
            pygame.draw.rect(self.screen, (15, 15, 23), uiRect)
            

            for i in range(self.texturSize):
                for j in range(self.texturSize):
                    self.screen.set_at((self.outputTexture[0]+i, self.outputTexture[1]+j), self.texture[i][j][1])
            
            ##############################################################
            ######################### EVENTS #############################
            ##############################################################

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    for i in range(self.texturSize):
                        for j in range(self.texturSize):
                            self.texture[i][j][1] = [0, 0, 0]
                            self.texture[i][j][2] = 1
                    filename = "image"
                    pygame.image.save(sub, os.path.join(self.appFolder, "saved_files/") + filename + '.png')
                    sys.exit(0)
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.buttons[0].collidepoint(event.pos):
                        self.logText = "Loading dataset ..."
                        self.Log("Loading dataset ...")
                        self.datasetDir = fd.askopenfile(title='Select dataset', initialdir=self.appFolder, filetypes=[('csv file', '.csv')])
                        print(self.datasetDir)
                        if(self.datasetDir != None):
                            self.logText = "Dataset loaded"
                            self.Log("Dataset loaded")
                        else:
                            self.logText = "No file selected"
                            self.Log("No file selected")

                    if self.buttons[1].collidepoint(event.pos):
                        try:
                            #model = buildModelT(self.datasetDir)
                            self.logText = "Loading model ..."
                            self.Log("Loading model ...")
                            modelPath = tk.filedialog.askopenfile(title='Open model file', initialdir=self.appFolder, filetypes=[('json file', '.json')])
                            loadModel(modelPath.name)
                            self.modelLoaded = True
                            self.logText = "Model loaded"
                            self.Log("Model loaded")
                        except:
                            print("error: File not selected.")
                            self.logText = "File not selected"
                            self.Log("File not selected")

                    if self.buttons[2].collidepoint(event.pos) and self.buttonActive[2]:
                        # model = buildModelT(self.datasetDir)
                        self.logText = "Preparing dataset..."
                        self.Log("Preparing dataset...")
                        testSize = 0.1
                        train_X, train_y, test_X, test_y = createDataSet(self.datasetDir, testSize) #### -------------- #####
                        if self.debugMode:
                           print("train_X shape: " + str(np.shape(train_X)))
                           print("train_y shape: " + str(np.shape(train_y)))
                           print("test_X shape: " + str(np.shape(test_X)))
                           print("test_y shape: " + str(np.shape(test_y)))
                        self.logText = "Dataset ready. Model training ongoing..."
                        self.Log("Dataset ready. Model training ongoing...")
                        trainModel(train_X, train_y, test_X, test_y, self.datasetDir) #### -------------- #####
                        self.trained = True
                        self.logText = "Model ready"
                        self.Log("Model ready")

                    if self.buttons[3].collidepoint(event.pos) and self.buttonActive[3]:
                        for i in range(self.texturSize):
                            for j in range(self.texturSize):
                                self.texture[i][j][1] = [0, 0, 0]
                                self.texture[i][j][2] = 1
                        filename = "image"
                        pygame.image.save(sub, os.path.join(self.appFolder, "saved_files/") + filename + '.png')


                if event.type == pygame.KEYDOWN:
                    if tbactive:
                        if event.key == pygame.K_RETURN:
                            print(filename)
                            filename = ''
                        elif event.key == pygame.K_BACKSPACE:
                            filename = filename[:-1]
                        else:
                            filename += event.unicode
                
                if event.type == pygame.VIDEORESIZE:
                    # There's some code to add back window content here.
                    surface = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

                        
                if event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] > self.w-self.uiw:

                    if saveButton.collidepoint(event.pos):
                        if(filename == ''):
                            filename = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                        pygame.image.save(sub, os.path.join(self.appFolder, "saved_files/") + filename + '.png')            


            if self.datasetDir == None:
                self.colors[2] = self.inactiveColor
                self.buttonActive[2] = False
            else:
                self.colors[2] = self.baseColor
                self.buttonActive[2] = True

            if pygame.mouse.get_pressed()[0]:
                for i in range(self.texturSize):
                    for j in range(self.texturSize):
                        # print(self.texture[i][j])
                        if pygame.Rect.collidepoint(self.texture[i][j][0], [pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]]):
                            if self.texture[i][j][1] != [255, 255, 255]:
                                
                                self.texture[i][j][1] = [255, 255, 255]
                                self.texture[i][j][2] = 0

                                # if (i > 0):
                                #     if (self.texture[i-1][j][1][0] < 215):
                                #         for c in range(3):
                                #             self.texture[i-1][j][1][c] += 40
                                #     else:
                                #         self.texture[i-1][j][1] = [255, 255, 255]
                                # if (j > 0):
                                #     if (self.texture[i][j-1][1][0] < 215):
                                #         for c in range(3):
                                #             self.texture[i][j-1][1][c] += 40
                                #     else:
                                #         self.texture[i][j-1][1] = [255, 255, 255]
                                # if (i < self.texturSize - 1):
                                #     if (self.texture[i+1][j][1][0] < 215):
                                #         for c in range(3):
                                #             self.texture[i+1][j][1][c] += 40
                                #     else:
                                #         self.texture[i+1][j][1] = [255, 255, 255]
                                # if (j < self.texturSize - 1):
                                #     if (self.texture[i][j+1][1][0] < 215):
                                #         for c in range(3):
                                #             self.texture[i][j+1][1][c] += 40
                                #     else:
                                #         self.texture[i][j+1][1] = [255, 255, 255]

                                filename = "image"
                                pygame.image.save(sub, os.path.join(self.appFolder, "saved_files/") + filename + '.png')
                                predict()
                            
            if(tbactive):
                pygame.draw.rect(self.screen, (146, 146, 178), textBox, width=2)
            else:
                pygame.draw.rect(self.screen, (72, 72, 82), textBox, width=2)
            filenamefont = font3.render(filename , True, (146, 146, 178))
            self.screen.blit(filenamefont,(self.w-self.uiw+10+4, self.h-50-50+2))

            pygame.draw.rect(self.screen, (72, 72, 82), saveButton, width=2)
            boxtext = font3.render('SAVE' , True, (146, 146, 178))
            self.screen.blit(boxtext,(saveButton[0]+saveButton[2]/2-boxtext.get_width()/2, saveButton[1]+saveButton[3]/2-boxtext.get_height()/2))


            for i in range(len(self.buttons)):
                pygame.draw.rect(self.screen, self.colors[i], self.buttons[i], width=2)
                text = self.font.render(self.texts[i], True, self.colors[i])
                self.screen.blit(text, (self.buttons[i].x+self.buttons[i].width/2-text.get_width()/2, self.buttons[i].y))

            pygame.draw.rect(self.screen, self.baseColor, predBox, width=2)
            if (len(self.supportVector) > 0):
                sortedSupport = self.supportVector.sort(reverse=True)
                for i in range(self.numberOfBest):
                    text = self.font.render(self.classes[self.supportVector.index(sortedSupport[i])] + sortedSupport[i], True, self.colors[i])
                    self.screen.blit(text, (self.textureSizeOnCanvas + 50, 25*i))

            for k in range(self.texturSize):
                for l in range(self.texturSize):
                    r = self.texture[k][l][0].copy()
                    color = self.texture[k][l][1]
                    pygame.draw.rect(self.screen, color, r, width=0)
            #for i in self.textures:
            pygame.draw.rect(self.screen, self.textureBorderColor, self.textures[0], width=1)
            
            self.Log(self.logText)

            pygame.display.flip()



if __name__ == "__main__":
    Game()
