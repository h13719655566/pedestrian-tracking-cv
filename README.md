# moving-object-tracker  
Python & OpenCV moving object detection and tracking system｜Python & OpenCV 移動物體偵測與追蹤系統

## Project Overview｜專案概述  
This project implements two tasks:  
- **Task One – Background Modelling (-b)**: Detect and classify moving objects (person, car, other) using MOG2 background subtraction, noise removal, connected component analysis, and contour classification. Outputs raw frame counts and statistics in console and shows original, background, foreground mask, and annotated frames in separate windows.  
- **Task Two – Pedestrian Detection & Tracking (-d)**: Use OpenCV DNN module with a pre-trained MobileNet SSD (COCO) to detect and track pedestrians, maintain consistent labels across frames, and highlight up to three closest persons. Displays original frames, detection overlays, tracked results, and closest-person-only view.  

本專案實作兩種模式：  
- **模式一 – 背景建模 (-b)**：使用 MOG2 背景減除，結合雜訊移除、連通元件分析與輪廓分類（person、car、other），在命令列輸出每幀物件統計，並分別開啟視窗顯示原始影像、背景估計、前景遮罩與標記後影像。  
- **模式二 – 行人偵測與追蹤 (-d)**：載入 MobileNet SSD (COCO) 模型，只偵測行人，維持物件在連續影格中的標籤一致，並標示最靠近鏡頭的前三名行人；在單一視窗依序顯示原始影像、偵測結果、追蹤結果及只顯示最近三位行人的畫面。

## Features｜核心功能  
- **Background Subtraction & Contour Classification (-b)**｜背景減除與輪廓分類（-b 模式）  
- **Pedestrian Detection & DNN-based Tracking (-d)**｜行人偵測與 DNN 追蹤（-d 模式）  
- Real-time visualisation of results｜即時結果可視化  
- Console output of object counts and statistics｜命令列輸出物件計數與統計  

## Directory Structure｜目錄結構  
```
.
├── movingObj.py        # Main script with both -b and -d modes｜主要程式：包含 -b 及 -d 模式
├── requirements.txt    # List of dependencies｜相依套件列表
└── README.md           # This readme file｜專案說明文件
```

## Dependencies｜相依套件  
- Python 3.7+｜Python 3.7 以上  
- numpy｜numpy  
- opencv-python｜opencv-python  

Install all dependencies via:｜透過以下指令安裝所有相依套件：  
```bash
pip install -r requirements.txt
```

## Usage｜使用說明  
執行程式時需指定模式與影片檔案，語法如下：  
```bash
python movingObj.py -b|-d path/to/video.mp4
```  
模式選項：  
- `-b`：執行 Task One，使用背景減除與輪廓檢測進行移動物體分類與計數（Background Subtraction & Contour Classification）。  
- `-d`：執行 Task Two，使用 DNN 模型（SSD MobileNet）進行行人偵測與追蹤（Pedestrian Detection & Tracking）。  

範例（Task One）：  
```bash
python movingObj.py -b samples/test_video.mp4
```  
範例（Task Two）：  
```bash
python movingObj.py -d samples/test_video.mp4
```

## Contributing｜貢獻指南  
Contributions are welcome! Please fork the repository and create a pull request.｜歡迎提出貢獻！請 fork 專案並發起 Pull Request。  
1. Fork the repository｜Fork 本專案  
2. Create a feature branch (`feature/YourFeature`)｜建立功能分支 (`feature/YourFeature`)  
3. Commit your changes｜提交修改  
4. Open a Pull Request｜發起 Pull Request  

## License｜授權條款  
This project is licensed under the MIT License.｜本專案採用 MIT 授權條款。  
