Description:
写真を読み込み、写真から顔・ランドマークを認識。
得られたランドマークを用いて、口紅メイクを施す。
なお、口紅の色はRGBA空間を使用。

Argument:
[Dlib HOG Detector model] [Image_Path] [Blue] [Green] [Red] [Transparency]
i.e.
shape_predictor_68_face_landmarks.dat　C:\03.jpg 30 0 255 1