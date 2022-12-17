import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


img = cv2.imread(r'D:\3_1\python project\Dataset\dataset1.png',1)
img2 = img.copy()
    
cv2.imshow("img",img)
demo = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
r,g,b = demo.shape

cv2.imshow('demo',demo)

key = np.random.randint(0,255,size=(r,g,b),dtype=np.uint8)
cv2.imshow('key',key)

encryption = cv2.bitwise_xor(demo,key)
cv2.imshow('encrypted image',encryption)


encryption_1 = cv2.cvtColor(encryption,cv2.COLOR_RGB2BGR)
key_1 = cv2.cvtColor(key,cv2.COLOR_RGB2BGR)



decryption = cv2.bitwise_xor(encryption_1,key_1)
cv2.imshow('decrypted image',decryption)


cv2.waitKey(-1)
cv2.destroyAllWindows()

##    EncryptionImg = np.zeros(img2.shape, np.uint8)
##    Encryption(img2,r,g,b,EncryptionImg)
##
##    cv2.imwrite(r"D:\3_1\python project\image123.png",EncryptionImg)
    

#decrypt method

def decryp(link,r,g,b):
    
    img = cv2.imread(link,1)
    DecryptionImg = np.zeros(img.shape, np.uint8)
    Decryption(img,r,g,b, DecryptionImg) 
    cv2.imwrite(r"D:\3_1\python project\decryp.png", DecryptionImg)




#================================

# function int to binary 8

def int2bin8(x):
    result = "";
    for i in range (8):
        y = x&(1)
        result += str(y)
        x=x>>1
    return result[::-1]


# fuction int to binary 16

def int2bin16(x):
    result = "";
    for i in range (16):
        y = x&(1)
        result += str(y)
        x=x>>1
    return result[::-1]


# function int to binary 32
def int2bin32(x):
    result = "";
    for i in range (32):
        y = x&(1)
        result += str(y)
        x=x>>1
    return result[::-1]

#  read the image
# read a password of size 10(normally phone number)

def convertcolormethod(x):
    if (x == 0):
        cv2.cvtColor(demo,cv2.COLOR_BGR2GRAY)
    elif (x == 1):
        cv2.cvtColor(demo,cv2.COLOR_BGR2RGB)
    elif (x == 3):
        cv2.cvtColor(demo,cv2.COLOR_BGR2BGRA)
    elif (x == 4):
        cv2.cvtColor(demo,cv2.COLOR_BGR2HLS)
    elif (x == 5):
        cv2.cvtColor(demo,cv2.COLOR_BGR2HSV)
    elif (x == 6):
        cv2.cvtColor(demo,cv2.COLOR_BGR2LAB)
    elif (x == 7):
        cv2.cvtColor(demo,cv2.COLOR_BGR2LUV)
    elif (x == 8):
        cv2.cvtColor(demo,cv2.COLOR_BGR2YUV)
    elif (x == 9):
        cv2.cvtColor(demo,cv2.COLOR_BGR2BGRA)
    else:
        print("error!")

def convertcolormethodback(x):
    if (x == 0):
        cv2.cvtColor(demo,cv2.COLOR_GRAY2BGR)
    elif (x == 1):
        cv2.cvtColor(demo,cv2.COLOR_RGB2BGR)
    elif (x == 3):
        cv2.cvtColor(demo,cv2.COLOR_BGRA2BGR)
    elif (x == 4):
        cv2.cvtColor(demo,cv2.COLOR_HLS2BGR)
    elif (x == 5):
        cv2.cvtColor(demo,cv2.COLOR_HSV2BGR)
    elif (x == 6):
        cv2.cvtColor(demo,cv2.COLOR_LAB2BGR)
    elif (x == 7):
        cv2.cvtColor(demo,cv2.COLOR_LUV2BGR)
    elif (x == 8):
        cv2.cvtColor(demo,cv2.COLOR_YUV2BGR)
    elif (x == 9):
        cv2.cvtColor(demo,cv2.COLOR_BGRA2BGR)
    else:
        print("error!")


# spltting and adding an image on top of the original

def arithmaticimage(img):
    image = img.copy();
    cv2.imshow("image",image)

    r,g,b = cv2.split(image)
    cv2.imshow("r",r)

    image2 = cv2.merge((b,g,r))
    cv2.imshow("image2",image2)

    image3 = cv2.add(image2,image)
    cv2.imshow("image3",image3)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()

# spltting and adding an image on top of the original reversal


def arithmaticimagereturn(img):
    image = img.copy()
    r,g,b = cv2.split(image)

    image2 = cv2.merge((b,g,r))
    image3 = cv2.substract(image2,image)
    cv2.imshow("image3",image3)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()




#create a mask with intensity transformations
# wrning is included
def mask():
    image = img.copy()
    cv2.imshow("image",image)

    image2 = cv2.add(image,100)
    cv2.imshow("image2",image2)

    image_3 = img.copy()
    image3 = cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)
    image4 = np.log((image3)+1)
    image_4 = np.array(image4, dtype = np.uint8)
    cv2.imshow("image4",image_4)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()



# method to take an input of 8 values this is the tunnel
def inputpassword():
    print("PASSWORD SIZE IS EIGHT")
    a = input("ENTER THE PASSWORD : ")

    array = list(a)
    
    return array

#contrast streching
def contraststreach():
    image = img.copy()
    cv2.imshow("img",image)

    input_max = np.max(image)
    print(input_max)

    input_min = np.min(image)
    print(input_min)

    output_max = 255
    output_min = 0

    output_img = ((image - input_min)*(output_max - output_min))/((input_max - input_min)+ output_min)
    cv2.imshow("output",output_img)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()



#histogram equalization
def histoeq():
    image = img.copy()
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original",image2)

    hist_eq = cv2.equalizeHist(image2)
    cv2.imshow("histogram equalization", hist_eq)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()
    
# histogram equalization with clashe
def histeqclashe():

    image = img.copy()
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original",image2)

    
    clashe = cv2.createCLAHE(clipLimit = 2.0 , tileGridSize = (8,8))
    gray_image_clashe = clashe.apply(image2)

    cv2.imshow("clashe image",gray_image_clashe)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()



#spatial filtering using custom filters
def filter2d():
    image = img.copy()
    kernal = np.ones((5,5),np.float32)/25
    cv2.imshow("original",image)

    result = cv2.filter2D(image,-1,kernal)
    cv2.imshow("result",result)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()


#unsharped masking
def unsharpmasking():
    image = img.copy()
    cv2.imshow("original",image)
    guass = cv2.GaussianBlur(image, (13,13), 0)
    mask = cv2.subtract(image, guass)

    sharpened_image = cv2.add(image, cv2.multiply(mask,0.95))
    cv2.imshow("sharpened_image",sharpened_image)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()


#===================================================

# encryption method
def Encryption(img,j0,g0,x0,EncryptionImg):
    x = img.shape[0]
    y = img.shape[1]
    c = img.shape[2]
    g0 = int2bin16(g0)
    for s in range(x):
        for n in range(y):
            for z in range(c):
                m = int2bin8(img[s][n][z])                   
                ans=""
                # print("ok")
                for i in range(8):
                    ri=int(g0[-1])                          
                    qi=int(m[i])^ri                         
                    xi = 1 - math.sqrt(abs(2 * x0 - 1))     
                    if qi==0:                                
                        xi=1-xi;
                    x0=xi                                    
                    t=int(g0[0])^int(g0[12])^int(g0[15])   
                    g0=str(t)+g0[0:-1]                      
                    ci=math.floor(xi*(2**j0))%2             
                    ans+=str(ci)
                re=int(ans,2)
                EncryptionImg[s][n][z]=re
    #arithmaticimage(EncryptionImg)
    #convertcolormethod(colormodel)
    
    
    
    


#decryption method
def Decryption(EncryptionImg, j0, g0, x0, DecryptionImg):
    #decrease arithmeticimage function
    #arithmaticimagereturn(EncrptionImg)
    #convert color model
    #convertcolormethodback(colormodel)
    x = EncryptionImg.shape[0]
    y = EncryptionImg.shape[1]
    c = EncryptionImg.shape[2]
    g0 = int2bin16(g0)
    for s in range(x):
        for n in range(y):
            for z in range(c):
                cc = int2bin8(img[s][n][z])
                ans = ""
                # print("no")
                for i in range(8):
                    xi = 1 - math.sqrt(abs(2 * x0 - 1))
                    x0 = xi
                    ssi = math.floor(xi * (2 ** j0)) % 2
                    qi=1-(ssi^int(cc[i]))
                    ri = int(g0[-1])
                    mi=ri^qi
                    t = int(g0[0]) ^ int(g0[12]) ^ int(g0[15])
                    g0 = str(t) + g0[0:-1]
                    ans += str(mi)
                re = int(ans, 2)
                DecryptionImg[s][n][z] = re








