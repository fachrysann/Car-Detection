# Near Real-Time Vehicle Detection from Jakarta’s Traffic CCTVs
## 📖 Introduction
This project uses the smallest YOLOv8 model to perform real-time vehicle detection on live public CCTV feeds across Jakarta. The model is lightweight, enabling fast predictions even on modest hardware. Although the accuracy may vary due to the trade-off for speed, the system shows strong potential for efficient and scalable traffic surveillance.<br>
🔧 Key Features:<br>
-🔴 Live streaming from public .m3u8 CCTV feeds (e.g. Bendungan Hilir, Gelora, Tomang, Jati Pulo)<br>
-🎯 Detection focused on car, motorcycle, bus, and truck<br>
-🧠 Model: YOLOv8n (nano) pretrained on COCO dataset<br>
-📊 Real-time vehicle counting and traffic status classification (No Traffic, Less Traffic, Crowded)<br>
-🌐 Simple Flask-based web interface with camera switching<br>
-🕒 Live timestamp overlay<br>
-💻 All processing runs locally (on your own device)<br>
-🌍 Public Access with Cloudflared:<br>
To allow others to view the real-time detection system without exposing your local IP or setting up a server, the app uses Cloudflared Tunnel. This makes the Flask app temporarily available over the internet via a secure trycloudflare.com link — perfect for sharing quick demos or remote testing.<br>
## 🧠 Results with YOLOv8nano
<img src="Content/GifExample.gif"/>
🛠️ This project uses the smallest YOLOv8 model to enable near real-time predictions. While the detections may not always be perfect due to the lightweight nature of the model especially on low-light condition, it demonstrates strong potential for efficient and scalable traffic monitoring solutions.
