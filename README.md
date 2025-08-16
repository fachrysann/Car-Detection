# 🚗 Near Real-Time Vehicle Detection from Jakarta’s Traffic CCTVs

## 📖 Introduction
This project demonstrates a **near real-time vehicle detection** system using live public CCTV feeds from across Jakarta. Leveraging the lightweight **YOLOv11s (Small)** model, the system is optimized for fast inference on modest hardware. While it trades off some accuracy for speed, it remains highly effective for scalable traffic monitoring and analysis.

---

## 🔧 Key Features

| Feature                               | Description                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------|
| 🔴 Live Streaming                     | Supports `.m3u8` CCTV feeds (e.g. Bendungan Hilir, Gelora, Tomang, Jati Pulo) |
| 🎯 Object Detection                   | Detects cars, motorcycles, buses, and trucks using **YOLOv8n (COCO pretrained)** |
| 📊 Traffic Status Classification     | Classifies traffic into `No Traffic`, `Less Traffic`, or `Crowded` based on vehicle count |
| 🌐 Web Interface                      | Flask-based interface with live stream, camera switching, and detection results |
| 🕒 Timestamp Overlay                  | Displays current time on the video feed                                    |
| 📈 Real-Time Chart                    | Line chart showing vehicle count per minute                                |
| 💻 Local Execution                    | Entire pipeline runs on your own device (no server required)               |
| 🌍 Cloud Access (Optional)           | Share your app with a public link using **Cloudflared Tunnel**             |

---

## 🧠 Results with YOLOv8n

<img src="Content/GifExamples.gif" alt="Detection Example"/>

This project uses the **smallest YOLOv11 model** to achieve high-speed predictions, suitable for near real-time analysis. While detection quality may slightly degrade in low-light conditions, the model still performs reliably for urban traffic monitoring tasks.

---

## 🧪 How to Run Locally
1. Install dependencies
<pre>pip install -r requirements.txt</pre>
2. Run the app
<pre>python app.py</pre>
3. Open in browser
<pre>http://localhost:5000</pre>
4. (optional) Expose it online using Cloudflared
<pre>cloudflared tunnel --url http://localhost:5000</pre>

## 📈 Results

The performance of the **XGBoost model with OSMnx-based route distance** is summarized below:

| Metric | Value |
|--------|-------|
| 🧮 **MAE**  | 1.918 |
| 📏 **RMSE** | 3.679 |
| 📊 **R²**   | 0.831 |

✅ The results show that incorporating **real road distances from OSMnx** significantly improves the model’s accuracy compared to using straight-line (Euclidean) distances.  
