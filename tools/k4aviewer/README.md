# K4AViewer

## Introduction

K4AViewer is a graphical utility that allows you to start Azure Kinect devices in various modes and visualize the data from the sensors.
To use, select a device from the list, click "Open Device", choose the settings you want to start the camera with, then click "Start".

## Usage Info

k4aviewer will try to detect if you have a high-DPI system and scale automatically; however, if you want to force it into or out of
high-DPI mode, you can pass -HighDPI or -NormalDPI to override that behavior.

```
k4aviewer.exe [-HighDPI|-NormalDPI]
```

# Important Notes

## Don't forget to copy `depthengine`
* from: Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin\depthengine_2_0.dll
* to: Azure-Kinect-Sensor-SDK\build\Win-x64-Debug-Ninja\bin\depthengine_2_0.dll

## Change OpenCV_DIR in CmakeSettings.json to your location