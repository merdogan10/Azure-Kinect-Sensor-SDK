// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CHARUCO_H
#define CHARUCO_H

// System headers
//
#include <iostream>
#include <fstream>

// Library headers
//
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include "linmath.h"

// Project headers
//
using namespace cv;
using namespace std;
namespace k4aviewer
{
void get_intrinsics(k4a_calibration_t calibration, Mat &camMatrix, Mat &distCoeffs)
{
    k4a_calibration_intrinsics_t intrinsics = calibration.color_camera_calibration.intrinsics;
    k4a_calibration_intrinsic_parameters_t intrinsic_params = intrinsics.parameters;

    camMatrix = (Mat_<float>(3, 3) << intrinsic_params.param.fx,
                 0,
                 intrinsic_params.param.cx,
                 0,
                 intrinsic_params.param.fy,
                 intrinsic_params.param.cy,
                 0,
                 0,
                 1);

    distCoeffs = (Mat_<float>(1, 8) << intrinsic_params.param.k1,
                  intrinsic_params.param.k2,
                  intrinsic_params.param.p1,
                  intrinsic_params.param.p2,
                  intrinsic_params.param.k3,
                  intrinsic_params.param.k4,
                  intrinsic_params.param.k5,
                  intrinsic_params.param.k6);
}

bool intrinsics_offline(const char *file_path, Mat &cameraMatrix, Mat &distCoeffs)
{
    k4a_playback_t playback_handle = NULL;
    if (k4a_playback_open(file_path, &playback_handle) != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to open recording\n");
        return false;
    }
    k4a_calibration_t calibration;
    k4a_result_t calibration_result = k4a_playback_get_calibration(playback_handle, &calibration);
    if (calibration_result != K4A_RESULT_SUCCEEDED)
    {
        printf("Failed to get calibration\n");
        k4a_playback_close(playback_handle);
        return false;
    }
    get_intrinsics(calibration, cameraMatrix, distCoeffs);

    k4a_playback_close(playback_handle);
    return true;
}
void MatrixMultiply(linmath::mat4x4 out, linmath::mat4x4 a, linmath::mat4x4 b)
{
    linmath::mat4x4 atmp;
    linmath::mat4x4 btmp;
    linmath::mat4x4_dup(atmp, a);
    linmath::mat4x4_dup(btmp, b);
    linmath::mat4x4_mul(out, a, b);
}
float Radians(const float angle)
{
    constexpr float pi = 3.14159265358979323846f;
    return angle / 180.f * pi;
}
void rotate_by_x(linmath::mat4x4 rotated, linmath::mat4x4 translation) {
    linmath::quat rotateQuat;
    linmath::vec3 rotateAxis{ 1.f, 0.f, 0.f };
    linmath::mat4x4 rotateMatrix;
    linmath::quat_rotate(rotateQuat, Radians(180), rotateAxis);
    linmath::mat4x4_from_quat(rotateMatrix, rotateQuat);
    MatrixMultiply(rotated, translation, rotateMatrix);
    //MatrixMultiply(rotated, rotateMatrix, translation);
}
class Charuco
{
public:
    void get_se3(linmath::mat4x4 se3)
    {
        linmath::mat4x4_dup(se3, m_se3);
    }
    void get_se3_inverse(linmath::mat4x4 se3_inverse)
    {
        linmath::mat4x4_dup(se3_inverse, m_se3_inverse);
    }
    Charuco(string input_path, string calib_file_name, bool is_quat, bool rotate_x = false) : m_rotate_x(rotate_x)
    {
        if (is_quat)
        {
            se3_from_quat(calib_file_name);
            return;
        }
        m_board_type = 4;
        init_board();
        intrinsics_offline(input_path.c_str(), m_cameraMatrix, m_distCoeffs);

        detect_pose(input_path);
        se3_from_rvec_tvec();
    }
    ~Charuco() = default;

private:
    void se3_from_quat(string file_name)
    {
        ifstream read(file_name);
        string empty;
        for (int i = 0; i < 10; i++)
            read >> empty;
        float qw, qx, qy, qz, tx, ty, tz;
        read >> qx >> qy >> qz >> qw;
        read >> empty >> empty;
        read >> tx >> ty >> tz;

        linmath::quat rotation_quat{qx,qy,qz,qw};
        linmath::mat4x4 transformation;
        linmath::mat4x4_from_quat(transformation, rotation_quat);
        transformation[3][0] = tx;
        transformation[3][1] = ty;
        transformation[3][2] = tz;

        if (m_rotate_x)
            rotate_by_x(m_se3, transformation);
        else
            linmath::mat4x4_dup(m_se3, transformation);

        linmath::mat4x4_invert(m_se3_inverse, m_se3);
    }

    void init_board()
    {
        switch (m_board_type)
        {
        case 5:
            m_dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
            m_board = aruco::CharucoBoard::create(5, 7, 0.04f, 0.02f, m_dictionary);
            break;
        case 14:
            m_dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_250);
            m_board = aruco::CharucoBoard::create(14, 9, 0.02f, 0.01556f, m_dictionary);
            break;
        case 4:
            m_dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);
            m_board = aruco::CharucoBoard::create(4, 3, 0.053f, 0.04f, m_dictionary);
            break;
        default:
            // cout << "Invalid Board" << endl;
            break;
        }
        m_params = aruco::DetectorParameters::create();
    }

    void detect_pose(string file_name)
    {
        VideoCapture inputVideo;
        inputVideo.open(file_name);

        inputVideo.grab();
        Mat image;
        inputVideo.retrieve(image);
        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners;
        aruco::detectMarkers(image, m_board->dictionary, markerCorners, markerIds, m_params);
        // if at least one marker detected
        if (markerIds.size() > 0)
        {
            vector<Point2f> charucoCorners;
            vector<int> charucoIds;
            aruco::interpolateCornersCharuco(
                markerCorners, markerIds, image, m_board, charucoCorners, charucoIds, m_cameraMatrix, m_distCoeffs);
            // if at least one charuco corner detected
            if (charucoIds.size() > 0)
            {
                aruco::estimatePoseCharucoBoard(
                    charucoCorners, charucoIds, m_board, m_cameraMatrix, m_distCoeffs, m_rvec, m_tvec);
            }
        }
    }

    void se3_from_rvec_tvec()
    {
        Mat rotation_cv;
        Rodrigues(m_rvec, rotation_cv);
        linmath::mat4x4 transformation;

        linmath::mat4x4_translate(transformation, (float)m_tvec(0), (float)m_tvec(1), (float)m_tvec(2));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                transformation[i][j] = (float)rotation_cv.at<double>(j, i); // linmath uses column index first

        if (m_rotate_x)
            rotate_by_x(m_se3, transformation);
        else
            linmath::mat4x4_dup(m_se3, transformation);

        linmath::mat4x4_invert(m_se3_inverse, m_se3);
    }

    bool m_rotate_x;
    Ptr<aruco::Dictionary> m_dictionary;
    Ptr<aruco::CharucoBoard> m_board;
    Ptr<aruco::DetectorParameters> m_params;
    int m_board_type;
    Mat m_cameraMatrix, m_distCoeffs;
    Vec3d m_rvec, m_tvec;
    linmath::mat4x4 m_se3, m_se3_inverse;
};


    
    } // namespace k4aviewer

#endif
