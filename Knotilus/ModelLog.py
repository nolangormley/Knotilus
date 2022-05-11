from datetime import datetime
import os.path

class ModelLog:

    def __init__(self, name_header = ''):
        self.name_header = name_header
        self.filename = self.name_header + datetime.now().strftime('_%Y-%m-%dT%H_%M_%S') + '.log'

        # If filename already exists, add to it
        while os.path.isfile(self.filename):
            self.filename += '_(1)' 

        self.openFile()

    def openFile(self):
        self.file = open(self.filename, 'w')
        self.file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\t' + self.name_header + ': Initializing Model\n')

    def logMessage(self, message):
        self.file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\t' + self.name_header + ':' + message + '\n')


        