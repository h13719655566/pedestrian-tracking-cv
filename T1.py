import cv2
import numpy as np

def movingObj(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("無法開啟影片檔案")
        return
    
    # 初始化背景建模器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        
        # 生成黑白遮罩
        fg_mask = bg_subtractor.apply(frame)
        background_frame = bg_subtractor.getBackgroundImage()
        
        # 使用形態學操作去除雜訊（僅針對左下角的黑白遮罩）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 使用連通組件分析來分離物體
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask_cleaned)
        
        # 創建一個全黑背景影像，僅顯示前景物體
        detected_objects = np.zeros_like(frame)
        person_count, car_count, other_count = 0, 0, 0
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 500:
                continue  # 忽略小面積物體
            
            # 使用寬高比進行分類
            aspect_ratio = w / float(h)
            if 0.4 < aspect_ratio < 1.5 and h > 50:
                person_count += 1
                label = "Person"
            elif aspect_ratio > 1.5:
                car_count += 1
                label = "Car"
            else:
                other_count += 1
                label = "Other"
            
            # 將每個連通組件範圍複製到 detected_objects，顯示原始顏色
            mask = np.zeros_like(fg_mask_cleaned)
            mask[labels == i] = 255  # 只顯示當前連通組件
            colored_component = cv2.bitwise_and(frame, frame, mask=mask)
            detected_objects = cv2.add(detected_objects, colored_component)
        
        # 組合成單一視窗顯示
        combined_window = np.zeros((960, 1280, 3), dtype=np.uint8)
        
        # 放置不同影像於單一視窗的四個區域
        combined_window[0:480, 0:640] = frame  # 左上角：原始影格
        if background_frame is not None:
            combined_window[0:480, 640:1280] = cv2.resize(background_frame, (640, 480))  # 右上角：背景影像（不進行形態學處理）
        combined_window[480:960, 0:640] = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)  # 左下角：黑白遮罩（經過形態學處理去除雜訊）
        combined_window[480:960, 640:1280] = detected_objects  # 右下角：僅顯示前景的彩色物體影像
        
        # 顯示合併的視窗
        cv2.imshow("Object Detection - Combined View", combined_window)
        
        # 在命令列輸出物體數量
        frame_count += 1
        print(f"Frame {frame_count:04d}: {num_labels - 1} objects ({person_count} persons, {car_count} cars, {other_count} others)")
        
        # 按 'q' 鍵退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 執行程式
video_file = "TownCentreXVID.avi"
movingObj(video_file)
