<h1 align="center">
Enhancing Event-Based Vehicle Detection: A Hybrid Spiking Neural Network and Vision Transformer Architecture
</h1>

<p align="center">
<b>Degree Project in Computer Science and Engineering (DA150X)</b>



KTH Royal Institute of Technology

Authors: Viggo Jahr & Axel Prander
</p>

## 📌 Project Overview
This repository contains the code for a Bachelor's thesis investigating a hybrid neuromorphic computer vision architecture. Traditional computer vision relies heavily on frame-based cameras and compute-heavy Convolutional Neural Networks (CNNs). To address energy and latency limitations, this project utilizes Event-based cameras and Spiking Neural Networks (SNNs).

To overcome the limitations of pure SNN architectures in capturing long-range spatial dependencies (such as distinguishing between visually similar vehicle classes like buses and trucks), this project introduces a Vision Transformer (ViT) as a global aggregation head. The detection task is framed as a heatmap regression problem, outputting a spatial map of Gaussian blobs corresponding to vehicle centers.

## 🏗️ Technical Foundation & Acknowledgements
This project is a direct continuation and expansion of previous research conducted at KTH Royal Institute of Technology. It builds upon the technical pipeline established by:

1. Emma Hagrot (2025): Original raw event data collection, cleaning, and formatting. (Original Repo)

2. Olof Eliasson & Tobias Persson (2025): Baseline pure-SNN architecture and the multi-class data preprocessing pipeline for traffic monitoring.

The data preprocessing scripts and data loading utilities in this repository are heavily based on their foundational work. Our novel contribution focuses on substituting the final fully-connected SNN layers with a ViT attention mechanism.

## ⚙️ Data Preprocessing Pipeline
The raw data consists of both a standard RGB camera recording and an asynchronous event stream captured via an Event Camera. The preprocessing pipeline aligns and formats this data for the model:

1. Video Processing: Downscale the frame-based video and split it into manageable clips (e.g., 1-minute intervals) using cut_video.py.

2. Event Binning: Accumulate asynchronous events into tensor representations using ~10ms time-windows via create_event_frames.py.

3. Label Generation: Utilize YOLOv11/12 on the standard RGB frames to generate bounding box labels for vehicles.

4. Label Transfer: Transfer the RGB bounding boxes to the event-camera perspective using a calculated homography matrix (transfer_labels.py).

5. Heatmap Generation: Transform the discrete bounding boxes into continuous Gaussian "blobs" representing object centers.

## 🧠 Model Architecture (Hybrid SNN-ViT)
The system is divided into two primary components:

* The Backbone (Feature Extractor): A Spiking Neural Network (or Convolutional) backbone that processes the 10ms event-frame bins to extract low-level spatial and temporal features efficiently.

* The Classification Head (Global Aggregation): A Vision Transformer (ViT) layer that replaces the standard SNN readout. It applies self-attention mechanisms to the feature maps to capture global context, outputting dense, class-specific heatmaps (e.g., 64x64xC).

These heatmaps are post-processed using peak extraction (local maxima) to map predicted object centers back to bounding boxes for final evaluation.

## 📦 Requirements
The main external packages and libraries required for this project include:

* PyTorch: Core deep learning framework for the Vision Transformer and model training.

* snnTorch / Norse: For compiling and simulating the Spiking Neural Network components.

* YOLO (Ultralytics): Required only if you are running the preprocessing pipeline from scratch to generate new labels from raw RGB video.

(Note: A complete requirements.txt file will be provided as the training environment is finalized.)

## 🚀 Usage & Installation
Setup instructions, training scripts, and evaluation commands will be added here as the codebase is actively developed during the DA150X course.
