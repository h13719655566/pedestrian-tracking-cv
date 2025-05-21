import cv2
import numpy as np


def movingObj(video_file):
    # 加載 DNN 模型和物體類別名稱
    model_weights = "frozen_inference_graph.pb"
    model_config = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"
    
    # 載入 DNN 模型
    net = cv2.dnn.readNet(model_weights, model_config, framework='Tensorflow')
    
    # 設定行人類別的 ID （MS COCO 的類別 ID 1 是 "person"）
    person_class_id = 1

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("無法開啟影片檔案")
        return

    frame_count = 0
    person_tracker = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        (h, w) = frame.shape[:2]

        # Step 1: 原始影像幀
        original_frame = frame.copy()
        
        # Step 2: 偵測行人並疊加邊界框
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        net.setInput(blob)
        detections = net.forward()

        detected_frame = frame.copy()
        labeled_frame = frame.copy()
        closest_frame = frame.copy()

        # 儲存行人資訊（標籤和距離估計）
        persons = []
        boxes = []
        confidences = []
        ids = []

        # 處理每個偵測結果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.25:  # 稍微降低置信度閾值
                class_id = int(detections[0, 0, i, 1])
                if class_id == person_class_id:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    boxes.append([startX, startY, endX - startX, endY - startY])
                    confidences.append(float(confidence))
                    ids.append(class_id)

        # 應用非極大值抑制 (NMS) 去除重疊偵測
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.3)

        # 檢查是否有任何 indices，避免 `flatten()` 出錯
        if len(indices) > 0:
            indices = indices.flatten()  # 若有數值，使用 flatten()
            
            # 定義最大框的寬度和高度閾值
            MAX_WIDTH = 200  # 可以根據畫面比例調整
            MAX_HEIGHT = 400

            for i in indices:  # 遍歷每個有效索引
                startX, startY, width, height = boxes[i]
                endX, endY = startX + width, startY + height
                centerX, centerY = (startX + endX) // 2, (startY + endY) // 2  # 計算中心點
                distance_estimate = endY - startY

                # 判斷是否已經追蹤該行人
                found_id = None
                min_distance = 50  # 定義一個距離閾值

                # 濾除過大的框
                if width > MAX_WIDTH or height > MAX_HEIGHT:
                    continue  # 忽略過大的框

                for pid, (prevX, prevY, _, _, prev_centerX, prev_centerY) in person_tracker.items():
                    dist = np.sqrt((centerX - prev_centerX) ** 2 + (centerY - prev_centerY) ** 2)
                    if dist < min_distance:
                        found_id = pid
                        person_tracker[pid] = (startX, startY, endX, endY, centerX, centerY)
                        break

                if found_id is None:
                    found_id = len(person_tracker) + 1
                    person_tracker[found_id] = (startX, startY, endX, endY, centerX, centerY)

                persons.append((found_id, startX, startY, endX, endY, distance_estimate))

                # 疊加邊界框（右上角區域），僅顯示框而不顯示ID標籤
                cv2.rectangle(detected_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)


        # Step 3: 標記和追蹤行人
        labeled_frame = frame.copy()  # 左下角影像使用此變數

        for (label_id, startX, startY, endX, endY, _) in persons:
            # 在框內顯示行人的 ID
            cv2.rectangle(labeled_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(labeled_frame, f"ID {label_id}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Step 4: 選取最接近的 3 名行人
        persons.sort(key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True)  # 使用框的面積排序
        for idx, (label_id, startX, startY, endX, endY, _) in enumerate(persons[:3]):
            cv2.rectangle(closest_frame, (startX, startY), (endX, endY), (0, 0, 255), 2) 

        # 顯示四個區域的影像
        combined_window = np.zeros((960, 1280, 3), dtype=np.uint8)
        combined_window[0:480, 0:640] = original_frame
        combined_window[0:480, 640:1280] = detected_frame
        combined_window[480:960, 0:640] = labeled_frame
        combined_window[480:960, 640:1280] = closest_frame

        cv2.imshow("Pedestrian Detection and Tracking - Combined View", combined_window)
        
        # 在命令列輸出每幀偵測到的行人數量
        frame_count += 1
        print(f"Frame {frame_count:04d}: {len(persons)} detected persons")

        # 按 'q' 鍵退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 執行程式
video_file = "TownCentreXVID.avi"
movingObj(video_file)
