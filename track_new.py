# -*- coding: utf-8 -*-
import numpy as np
import cv2
from playsound import playsound


#import sys 




def sound():
    playsound('pokemon_get.mp3')





if __name__ == '__main__': # メイン文
        count = 0

        fullbody_detector = cv2.CascadeClassifier("/usr/share/opencv/haarcascades//haarcascade_upperbody.xml")



        template_img = cv2.imread("template.jpeg")
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        # テンプレートマッチング画像の高さ、幅を取得する
        h_tem, w_tem = template_gray.shape
        
        filepath = "bokemon_go.mov"
        
        cap = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(filepath)
        
        before = None
        while True:
    
            #  OpenCVでWebカメラの画像を取り込む
    
            ret, frame = cap.read()

            #gray_mask = white_detect(frame)
            gray_mask = frame
            
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_BGR2GRAY)
            
            body = fullbody_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(40, 40))
            # スクリーンショットを撮りたい関係で1/4サイズに縮小
            #frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))
            # 加工なし画像を表示する
            #cv2.imshow('Raw Frame', frame)

            # 取り込んだフレームに対して差分をとって動いているところが明るい画像を作る
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if before is None:
                before = gray.copy().astype('float')
                continue
            # 現フレームと前フレームの加重平均を使う
            cv2.accumulateWeighted(gray_mask, before, 0.5)
            mdframe = cv2.absdiff(gray_mask, cv2.convertScaleAbs(before))
            ksize = 3
            mdframe = cv2.medianBlur(mdframe,ksize)
            
            # 処理対象画像に対して、テンプレート画像との類似度を算出する
            res = cv2.matchTemplate(mdframe, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # 類似度の高い部分を検出する
            threshold = 0.65
            loc = np.where(res >= threshold)


            # 動いているところが明るい画像を表示する
            cv2.imshow('MotionDetected Frame', mdframe)
    
            # 動いているエリアの面積を計算してちょうどいい検出結果を抽出する
            thresh = cv2.threshold(mdframe, 3, 255, cv2.THRESH_BINARY)[1]
            # 輪郭データに変換しくれるfindContours
            image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            target = contours[0]
            for cnt in contours:
                #輪郭の面積を求めてくれるcontourArea
                area = cv2.contourArea(cnt)
                if max_area < area and area < 10000:
                    max_area = area;
                    target = cnt
                    
            # 動いているエリアのうちそこそこの大きさのものがあればそれを矩形で表示する
            if body != () or count >= 1:
                if max_area <= 1 and count == 0:
                    areaframe = frame
                    cv2.putText(areaframe, 'not detected', (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3, cv2.LINE_AA)
                else:
                    #print("body!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        #img = np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]),np.uint8)
                        #cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,0),2)
                        #count = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
                        #cv2.fillPoly(img, pts=[count], color=(0,0,0))
                        # 諸般の事情で矩形検出とした。
                        x,y,w,h = cv2.boundingRect(target)
                        #dx = np.minimum(x+w, x_b+w_b) - np.maximum(x, x_b)
                        #dy = np.minimum(y+h, y_b+h_b) - np.maximum(y, y_b)
                        #IoU = dx*dy / (w*h)
                        #dis = np.sqrt((x + w/2 - x_b - w_b/2)**2 + (y + h/2 - y_b - h_b/2)**2)
                        
                        #if dx > 0 and dy > 0 and IoU < 0.01:
                        #if dx<0 or dy<0:
                        if np.any(res >= threshold) == True or count >= 1:
                            count += 1
                            # 検出した部分に赤枠をつける
                            areaframe = frame
                            for pt in zip(*loc[::-1]):
                                cv2.rectangle(areaframe, pt, (pt[0] + w_tem, pt[1] + h_tem), (0, 0, 255), 2)
                    

                            areaframe = cv2.rectangle(areaframe,(x,y),(x+w,y+h),(0,255,0),2)
                    
                            cv2.putText(areaframe, 'detected', (0,300), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3, cv2.LINE_AA)
                            if count == 1:
                                playsound("pokemon_get.mp3")
                                break
                
                            #img = np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]),np.uint8)
                            #cv2.rectangle(img, (x, y),(x+w, y+h),(0,255,0),2)
                            #count = np.array([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
                            #cv2.fillPoly(img, pts=[count], color=(0,0,0))
                    
                            #bodyframe = cv2.rectangle(frame,(x_b,y_b),(x_b+w_b,y_b+h_b),(0,255,0),2)

                            #cv2.imshow('BodyDetected Area Frame', bodyframe)
                        else:
                            areaframe = frame
            else:
                areaframe = frame
                cv2.putText(areaframe, 'no human', (0,200), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, cv2.LINE_AA)


            cv2.imshow('MotionDetected Area Frame', areaframe)
            # キー入力を1ms待って、k が27（ESC）だったらBreakする
            k = cv2.waitKey(1)
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
            
        #p= Process(target=sound)
        #p.start()
        #p.join()


        while(cap2.isOpened()):
            ret2, frame2 = cap2.read()
        
            #cv2_putText_2(frame2, 'ボッチャマ　CP ???', (0,200),  'ipagp.ttf', 3, (0,0,255))
            frame3 = frame2
            #frame3 = cv2.resize(frame3, (int(frame3.shape[1]), int(frame3.shape[0])))
            #print(frame2.shape)
            cv2.imshow("Bokemon", frame3)
        
            k = cv2.waitKey(1)
            if k == 27:
                break

        # キャプチャをリリースして、ウィンドウをすべて閉じる
        cap2.release()
        cv2.destroyAllWindows()

        
