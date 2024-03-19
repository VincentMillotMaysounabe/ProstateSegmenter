"""
author : VMI
creation date : 17/10/2023
description : to store & load large numbers of dicom images from ProstateX dataset
"""
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os


class ProstateXImageStorage:
    def __init__(self, path):
        self._path = path
        self._nbSubjects = 0
        self._examListPath = []
        self._segListPath = []
        self._outputImg = None
        self._outputSeg = None

    def _dataAcces(f):
        """
        wrapper to make sure the data access is correct
        :param f:
        :return:
        """
        def decorateur(self, id, nb = 0):#id: int, nb: int = 0):
            Launch = True
            if id > len(self._examListPath):
                print("Error <dataAccess> :: Passed id out of range")
                Launch = False

            if Launch:
                f(self, id, nb)

        return decorateur

    def loadImages(self) -> None:
        """
        Discovering every .nii in path
        :return:
        """

        examFolderIdentifier = 't2tsetra'
        segmentationFolderIdentifier = 'Segmentation'

        for root, _, files in os.walk(self._path):
            if examFolderIdentifier in root.split('/')[-1].split('\ '[0])[-1]:
                self._examListPath.append(root)
                self._nbSubjects += 1

        for root, _, files in os.walk(os.path.join(self._path)):
            if segmentationFolderIdentifier in root.split('/')[-1].split('\ '[0])[-1]:
                self._segListPath.append(os.path.join(root, files[0]))

        listsubj = [self._examListPath[i].split('\ '[0])[-3] for i in range(len(self._examListPath))]
        k = 0
        for i in range(len(listsubj)):
            if listsubj[i] in listsubj[i+1:]:
                self._examListPath.pop(k+i)
                self._nbSubjects -= 1
                k -= 1

        print(f"Found {self._nbSubjects} subjects and {len(self._segListPath)} segmentations")

    @_dataAcces
    def showHeader(self, id: int) -> None:
        """
        Show header of the data pointed by the id
        :param id: id of data
        :return:
        """
        path = self._examListPath[id]
        for root, _, files in os.walk(path):
            print(f"root : {root}")
            nbfiles = len(files)
            print(f"Found {nbfiles} dcm files for this id\n")
            lastFile = pydicom.dcmread(os.path.join(root, files[-1]))
            print("Metadata from last file :\n")
            print(lastFile.file_meta)
            print(f"Pixel array shape : {lastFile.PixelData.shape()}")

    @_dataAcces
    def getData(self, id: int, nb):
        """
        Return the data pointed by id
        :param id: id of data to get loaded
        :param category:
        :return:
        """
        path = self._examListPath[id]
        allPixelArray = []
        for root, _, files in os.walk(path):
            for file in files:
                curFile = pydicom.dcmread(os.path.join(root, file))
                allPixelArray.append(curFile.pixel_array.copy())
        allPixelArray = np.array(allPixelArray)
        self._outputImg = allPixelArray

        segFile = pydicom.dcmread(self._segListPath[id])
        outputSeg = []
        Nimg = len(segFile.pixel_array)//4
        for i in range(Nimg):
            outputSeg.append(segFile.pixel_array[-i,:,:] + segFile.pixel_array[-(i+Nimg),:,:]
                             + segFile.pixel_array[-(i +2*Nimg),:,:] + segFile.pixel_array[-(i +3*Nimg),:,:])
        outputSeg = np.array(outputSeg)
        self._outputSeg = outputSeg

    @_dataAcces
    def ShowImage(self, id, nb):
        """
        Show the image pointed by id and nb
        :param id: id of data to get loaded
        :param nb: number of the image to display
        :return:
        """
        path = self._examListPath[id]
        for root, _, files in os.walk(path):
            if len(files) > nb >= 0:
                file = pydicom.dcmread(os.path.join(root, files[nb]))
                plt.imshow(file.pixel_array, cmap='gray')
            else:
                print("Error <ShowImage> : image doesn't exist")

    @_dataAcces
    def ShowSegmentation(self, id : int, nb: int) -> None:
        """
        Use matplotlib.pyplot to display the N° nb segmentation from subject N°id
        :param id: id of data to get loaded
        :param nb: number of the image to display
        """
        if len(self._segListPath) > id >= 0:
            fileSeg = pydicom.dcmread(self._segListPath[id])
            if len(fileSeg.pixel_array) > nb > 0:
                plt.imshow(fileSeg.pixel_array[nb, :, :], cmap='gray')
            else:
                print("Error <ShowSegmentation> : segmentation doesn't exist")
        else:
            print("Error <ShowSegmentation : file doesn't exist>")

    @_dataAcces
    def ShowImgWithContours(self, id: int, nb: int) -> None:
        """
        Use matplotlib.pyplot to display the N° nb segmented part of the scan from subject N°id
        :param id: id of data to get loaded
        :param nb: number of the image to display
        """
        self.getData(id, nb)
        mixed = self._outputImg*self._outputSeg
        plt.imshow(mixed[nb], cmap='gray')

    @property
    def nbSubjects(self) -> int:
        return self._nbSubjects

    @property
    def nbSegmentation(self) -> int:
        return len(self._segListPath)

    @property
    def outputImg(self) -> list:
        return self._outputImg

    @property
    def outputSeg(self) -> list:
        return self._outputSeg

