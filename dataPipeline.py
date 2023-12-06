import pandas as pd
import numpy as np
import librosa as lb
import os
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import random 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import imageUtils as IU
import tensorflow_io as tfio

#TO DO: implement functions to load,save, store and acces data conveniently

class track:
    title="null"
    mSpGram="null"
    oneHotLabel="null"
    def __init__(self,data,frequency,label,id):
        self.label=label
        self.rawData=data
        self.frequency=frequency
        self.id=id
        return
    def cut(self,duration): #if duration is a divisor of the original lenght ignore the last element of the return
        sliceLen = int(duration * self.frequency)
        nSlices = len(self.rawData) // sliceLen
        self.cutTrack= []

        for i in range(nSlices):
            start = i * sliceLen
            end = start + sliceLen
            audio_slice = self.rawData[start:end]
            self.cutTrack.append(audio_slice)

    # Handle the remaining portion (last slice)
        remaining = self.rawData[nSlices * sliceLen:]
        if len(remaining) > 0:
            self.cutTrack.append(remaining)
            
        return self.cutTrack
    
    def toMelSpectroGram(self,mels=128):
        np.random.seed(0)  # Set a specific seed for NumPy
        lb.display.__random_state__ = np.random.RandomState(0)
        self.mSpGram= tfio.audio.melscale(tfio.audio.spectrogram(input=self.rawData,nfft=2048,window=2048,stride=512),mels=mels,rate=self.frequency,fmin=0,fmax=8000)
        #self.mSpGram=lb.power_to_db(lb.feature.melspectrogram(y=self.rawData,
         #                          sr=self.frequency,
          #                         n_mels=mels),ref=np.max)
        return self.mSpGram

    
    def len(self):
        self.lenght=len(self.rawData)/self.frequency
        return self.lenght
    
    def save(self,path="",mode="data"): #either "data" or "mel", one saves all the data the other just the mel spectrogram
        if mode=="data":
            name=f"{self.title}.txt"
            with open (os.path.join(path,name), mode="w",encoding="utf-8") as f:
                toWrite=f"{self.rawData}|{self.frequency}|{self.id}|{self.label}|{self.title}"
                f.write(toWrite)
            return os.path.join(path,name)
        elif mode=="mel": #obsolete we don't need it anymore
            if type(self.mSpGram)!=str:
                name=f"{self.title} melSpectrogram"
                p=f"{os.path.join(path,name)}.png"
                plt.figure(figsize=(10, 4))
                lb.display.specshow(lb.power_to_db(self.mSpGram, ref=np.max), y_axis='mel', x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title(name)
                plt.savefig(p)
                plt.close()
            return p 


def stdGenre(string):
    s=list(string)
    out=""
    for char in s:
        if char.isalpha():
            out+=f"{char.lower()}"
    return out

def loadAndParse(pathList):
    
    n=0
    trackList=[]
    for ind,elem in enumerate(pathList):
        try:
            #print(elem)
            #print("here?")
            id3 = ID3(elem)
            genre = id3.get('TCON', ['Unknown'])[0] 
            genre=stdGenre(genre)
            title = id3.get('TIT2', ['Unknown'])[0]
            title=stdGenre(title)
            data,fr=lb.load(elem)
            idSong=n
            n+=1
            t=track(data=data,frequency=fr,label=genre,id=idSong)
            t.title=title
            #saveGenre("assets\genres.txt",genre)
            trackList.append(t)
            
        except Exception as e:
            print(f"An error occurred while processing file {elem}: {e}")
    #print("here1")
    #resQueue.put(trackList)
    #print("here")

    return trackList

def loadAndParseFromTxt(path,mode="list"): #can either be "list" or "file" default is list
    if mode=="file":
        with open(path, mode="r",encoding="utf-8") as f:
            for line in f:
                string=line.split("|")
            t=track(data=string[0],frequency=string[1],id=string[2],label=string[3])
            t.title=string[4]
        return t     
    else:
        trackList=[]
        for ind,elem in enumerate(path):
            file=os.path.join(elem)
            with open(file, mode="r",encoding="utf-8") as f:
                for line in f:
                    string=line.split("|")
                t=track(data=string[0],frequency=string[1],id=string[2],label=string[3])
                t.title=string[4]
                trackList.append(t)
        return trackList




def sortTracks(trList,targetPath):
    genres=[]
    for ind, elem in enumerate(trList):
        pathName=os.path.join(targetPath,elem.label)
        if elem.label not in genres:
            genres.append(elem.label)
            try:
                os.mkdir(pathName)
            except:
                pass 
        elem.save(path=pathName, mode="data")
    return len(genres)

def multiProcessLoad(path,n=1,mode="mp3"): #optimal value is 8 (probably just half the core of the CPU)
    #resQueue = multiprocessing.Queue()
    res=[]
    allFiles=[os.path.join(path,elem) for elem in os.listdir(path)]
    size = len(allFiles) // n
    splitList=[allFiles[i:i + size] for i in range(0, len(allFiles), size)] 

    if mode=="txt":
        results = Parallel(n_jobs=n, backend='loky', prefer='processes', return_as="list")(delayed(loadAndParseFromTxt)(fileList) for fileList in splitList)

    else:
        results = Parallel(n_jobs=n, backend='loky', prefer='processes', return_as="list")(delayed(loadAndParse)(fileList) for fileList in splitList)
    
    #results=filterGenres(results,50)
    
    if mode=="txt":
        for l in results:
            res+=l
    else:
        for l in results:
            for elem in l:
                saveGenre("assets\genres.txt",elem.label)
                res.append(elem)
    return res


def saveGenre(path,genre):
    towrite=[]
    genres=[]
    alreadySeen=[]
    genres.append(genre)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            linelist=line.split("|")
            for n in range(int(linelist[1])):
                genres.append(linelist[0])
        f.close()
    for gen in genres:
        if gen not in alreadySeen:
            alreadySeen.append(gen)
            towrite.append(f"{gen}|{genres.count(gen)} \n")
        else: 
            pass
    with open(path, "w", encoding="utf-8") as f:
        towrite.sort(key= lambda k: k[0])
        for elem in towrite:
            f.write(elem)
        f.close()
    return  


def filterGenres(trList,n=50): # to test
    seen=[]
    for elem in trList:
        if elem not in seen:
            fr=trList.count(elem.label)


def oneEncode(trackList,n):
    enc=[0 for x in range(n)]
    seen=[]
    n=0
    for ind,elem in enumerate(trackList):
        if elem.label not in seen:
            enc[n]=1
            seen.append(elem.label)
            elem.oneHotLabel=enc
            enc[n]=0
            n+=1    
    return trackList


def normalizeDim(dim,arr):
    if dim==arr.shape:
        return arr
    else:
        if arr.shape[0]>dim[0]:
            c=np.abs(arr.shape[0]-dim[0])
            arrM= arr[:-c, :]
        return arrM
    
def multiProcessTask(task,inputList,n): #parallelize any task that can be done over a list
    res=[]
    size = len(inputList) // n
    splitList=[inputList[i:i + size] for i in range(0, len(inputList), size)]
    results = Parallel(n_jobs=n, backend='loky', prefer='processes', return_as="list")(delayed(task)(inpList) for inpList in splitList)
    for l in results:
            for elem in l:
                res.append(elem)
    return res 

def process(trList):
    for elem in trList:
        elem.toMelSpectroGram()
        

    return trList


def preProcess(folderPath,split=0.2,nGenres=20):
    nCores=8
    #load the data
    trackList=multiProcessLoad(folderPath,n=nCores) #load the data
    
    #filter the data based on the top 50 genres

    #process the data
    trackList=oneEncode(trackList,nGenres)
    processed=multiProcessTask(process,trackList,n=nCores)
    
    #split the data
    sP=len(processed)*split
    random.shuffle(processed)
    trainingX=[elem.smpGram for elem in processed[sP:]]
    trainingY=[elem.onHotLabel for elem in processed[sP:]]
    validationX=[elem.smpGram for elem in processed[:sP]]
    validationY=[elem.onHotLabel for elem in processed[:sP]]
    data=[trainingX,trainingY,validationX,validationY]
    return data







if __name__ == '__main__':

    def main():
        dirPath="path_to_data"
        tList=multiProcessLoad(dirPath,n=1)
        for t in tList:
            f=tList[0].toMelSpectroGram(mels=128)
            #im= tList[10].save(mode="mel")
            #m=IU.preProcess(im)
            print(f.shape)
            #print(t.title)
        return
    


    main()