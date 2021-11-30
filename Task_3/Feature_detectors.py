import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix as c_m



path_data = 'Data'
path_etalons = 'Etalons'

### Create feature detectors
sift = cv2.xfeatures2d.SIFT_create() 

orb = cv2.ORB_create()

star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()


### Import Etalons
classNames = []
folder=os.listdir(path_etalons)
images=[]
for i in range(len(folder)):
    images.append([])
print('Total Classes Detected', len(folder))
nbClass=-1;
for type in folder:
    nbClass=nbClass+1
    label=os.path.splitext(type)[0]
    classNames.append(label.split('_')[1])
    imgToImport=os.listdir(f'{path_etalons}/{type}')
    for im in imgToImport:
        imgCur = cv2.imread(f'{path_etalons}/{type}/{im}',0)
        images[nbClass].append(imgCur)
print(classNames)

maxIndex=0
folder=os.listdir(path_data)
for im in folder:
    name=os.path.splitext(im)[0]
    if int(name)>maxIndex:
        maxIndex=int(name)



###Use to find descriptors of etalons
def findDes(images,algo):
    desList=[]
    for i in range(len(images)):
        desList.append([])
    for typeNb in range(len(images)):
        for img in images[typeNb]:
            if algo==sift or algo==orb:
                kp,des = algo.detectAndCompute(img,None)
            if algo==brief:
                kp = star.detect(img,None)
                kp,des = brief.compute(img,kp)
            desList[typeNb].append(des)
    return desList


###Use to find match between image and etalons
def findID(img, desList, algo):

    if algo==orb:
        kp2, des2 = orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
   
    elif algo==sift:
        kp2, des2 = sift.detectAndCompute(img,None)
        bf = cv2.BFMatcher()

    elif algo==brief:
        kp2 = star.detect(img,None)
        kp2,des2 = brief.compute(img,kp2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    matchList = [0] * len(desList)
    finalVal = -1
    try:
        for typeNb in range(len(desList)):
            for des in desList[typeNb]:

                if algo==sift:
                    matches = bf.knnMatch(des,des2,k=2)
                    good = []
                    for m,n in matches:
                        if m.distance < 0.75*n.distance:
                            good.append([m])
                    matchList[typeNb] = matchList[typeNb] + len(good)

                elif algo==orb or algo==brief:
                    matches = bf.match(des,des2)
                    matches = sorted(matches, key = lambda x:x.distance)
                    good = []
                    for m in matches:
                        if(m.distance<300):
                            good.append([m])
                    matchList[typeNb] = matchList[typeNb] + len(good)

                else:
                    print("Wrong entrie")             
    except:
        pass

    if len(matchList)!=0:
        finalVal = matchList.index(max(matchList))
    
    return matchList


###Use to read the water plant type from the label file
def readLabel(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for name in root.iter('name'):
        return name.text

    
###Use to gather the real type of the image (for metrics)
def realMatch(path_data):
    folder=os.listdir(path_data)
    real = [None]*maxIndex

    for file in folder:
        name=os.path.splitext(file)[0]
        type=os.path.splitext(file)[1]

        if(type=='.xml'):
            real[int(name)-1]=classNames.index(readLabel(f'{path_data}/{file}'))
        
    return real


###Use to predict the type of water plant
def predictMatch(desList, algo, path_data):
    folder=os.listdir(path_data)
    predict = [None]*maxIndex
    overlaps = []

    for file in folder:
        name=os.path.splitext(file)[0]
        type=os.path.splitext(file)[1]
    
        if(type=='.jpg'):
            im = cv2.imread(f'{path_data}/{file}',0)
            matchList = findID(im,desList,algo)
            
            finalVal = matchList.index(max(matchList))
            if max(matchList)>300 and algo==sift:
                overlaps.append(int(name))
            predict[int(name)-1]=finalVal

    if algo==sift:
        return predict, overlaps
     
    return predict


###Main
desList_s = findDes(images,sift)
desList_o = findDes(images,orb)
desList_b = findDes(images,brief)
real = realMatch(path_data)

print("\nComputing sift ...")
predict_sift,overlaps = predictMatch(desList_s, sift, path_data)
print("Computing orb ...")
predict_orb = predictMatch(desList_o, orb, path_data)
print("Computing brief ...")
predict_brief = predictMatch(desList_b, brief, path_data)


r_s = []
r_o = []
r_b = []
p_s = []
p_o = []
p_b = []

print("-------------------------------")
print("Prediction:\n")
print("Image\tSift\tOrb\tBrief\tReal")
for i in range(maxIndex):
    if real[i]!=None:
        print(str(i+1)+'\t'+str(predict_sift[i])+'\t'+str(predict_orb[i])+'\t'+str(predict_brief[i])+'\t'+str(real[i]))

        ###Remove measurement overlap for metrics
        if (i+1 in overlaps and real[i]!=predict_sift[i]) or (i+1 not in overlaps):
            r_s.append(real[i])
            p_s.append(predict_sift[i])
        if (i+1 in overlaps and real[i]!=predict_orb[i]) or (i+1 not in overlaps):
            r_o.append(real[i])
            p_o.append(predict_orb[i])
        if (i+1 in overlaps and real[i]!=predict_brief[i]) or (i+1 not in overlaps):
            r_b.append(real[i])
            p_b.append(predict_brief[i])
        
       

print("-------------------------------\n")

print("Metrics:")

print("\nSift")
print(confusion_matrix(r_s, p_s))
cm = c_m(y_target=r_s, y_predicted=p_s)
fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_absolute=False, show_normed=True, figsize=(2, 2), class_names=classNames)
plt.show()
print("Recall : "+str(recall_score(r_s, p_s, average='micro')))
print("Precision : "+str(precision_score(r_s, p_s, average='micro')))
print("F1_score : "+str(f1_score(r_s, p_s, average='micro')))

print("\nOrb")
print(confusion_matrix(r_o, p_o))
cm = c_m(y_target=r_o, y_predicted=p_o)
fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_absolute=False, show_normed=True, figsize=(2, 2), class_names=classNames)
plt.show()
print("Recall : "+str(recall_score(r_o, p_o, average='micro')))
print("Precision : "+str(precision_score(r_o, p_o, average='micro')))
print("F1_score : "+str(f1_score(r_o, p_o, average='micro')))

print("\nBrief")
print(confusion_matrix(r_b, p_b))
cm = c_m(y_target=r_b, y_predicted=p_b)
fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_absolute=False, show_normed=True, figsize=(2, 2), class_names=classNames)
plt.show()
print("Recall : "+str(recall_score(r_b, p_b, average='micro')))
print("Precision : "+str(precision_score(r_b, p_b, average='micro')))
print("F1_score : "+str(f1_score(r_b, p_b, average='micro')))



