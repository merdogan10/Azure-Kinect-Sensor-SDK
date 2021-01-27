// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CALIBRATION_H
#define CALIBRATION_H

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
/**
* \brief degree to radian converter
*/
float Radians(const float angle)
{
    constexpr float pi = 3.14159265358979323846f;
    return angle / 180.f * pi;
}
/**
* \brief helper function to rotate x-dimension by 180 degrees.
* 
* mainly used to convert VICON left-handed axis to Azure Kinect right-handed axis
*/
void rotate_by_x(linmath::mat4x4 rotated, linmath::mat4x4 translation) {
    linmath::quat rotateQuat;
    linmath::vec3 rotateAxis{ 1.f, 0.f, 0.f };
    linmath::mat4x4 rotateMatrix;
    linmath::quat_rotate(rotateQuat, Radians(180), rotateAxis);
    linmath::mat4x4_from_quat(rotateMatrix, rotateQuat);
    linmath::mat4x4_mul(rotated, translation, rotateMatrix);
}
/**
 * \brief helper function to reverse x-dimension.
 * \param multiply_by_right indicates which side to be multiplied
 */
void reverse_x(linmath::mat4x4 result, linmath::mat4x4 input, bool multiply_by_right = true) {
    linmath::mat4x4 transform_x, temp_input;
    linmath::mat4x4_dup(temp_input, input);
    linmath::mat4x4_identity(transform_x);
    transform_x[0][0] = -1;
    if (multiply_by_right)
        linmath::mat4x4_mul(result, temp_input, transform_x); // Quaternion, ICP
    else
        linmath::mat4x4_mul(result, transform_x, temp_input); // Charuco
}
enum class CalibrationType
{
    Quaternion,
    Charuco,
    ICP
};
class Calibration
{
public:
    void get_extrinsics(linmath::mat4x4 se3)
    {
        linmath::mat4x4_dup(se3, m_extrinsics);
    }
    void get_se3(linmath::mat4x4 se3)
    {
        linmath::mat4x4_dup(se3, m_se3);
    }
    void get_se3_inverse(linmath::mat4x4 se3_inverse)
    {
        linmath::mat4x4_dup(se3_inverse, m_se3_inverse);
    }
    /**
    * \brief Main calibration pipeline
    * \param input_video
    * \param world2camera_file path to world2camera calibration file
    * \param icp_file path to icp calibration file
    * \param calibration_type can be only element of enum class CalibrationType
    */
    Calibration(string input_video, string world2camera_file, string icp_file, CalibrationType calibration_type)
    {
        switch (calibration_type)
        {
        case k4aviewer::CalibrationType::Quaternion:
            se3_from_quat(world2camera_file);
            break;

        case k4aviewer::CalibrationType::Charuco:
            init_board();
            intrinsics_extrinsics(input_video.c_str(), m_cameraMatrix, m_distCoeffs, m_extrinsics);
            detect_pose(input_video);
            se3_from_rvec_tvec();
            break;

        case k4aviewer::CalibrationType::ICP:
            se3_from_icp(icp_file);
            break;

        default:
            throw std::runtime_error("Selected an unsupported calibration type!");
            break;
        }

        linmath::mat4x4_invert(m_se3_inverse, m_se3);
    }
    ~Calibration() = default;

private:
    /**
     * \brief generates se3 from icp transformation file
     * \param file_name path to icp file
     */
    void se3_from_icp(string file_name) {
        ifstream read(file_name);
        float x;
        linmath::mat4x4 transformation;
        for (int i = 0; i <= 3; i++)
            for (int j = 0; j <= 3; j++)
            {
                read >> x;
                if (j == 3 && i != 3)
                    x /= 1000; // given in mm by Cloud Compare
                transformation[j][i] = x;
            }
        linmath::mat4x4_dup(m_se3, transformation);

        reverse_x(m_se3, m_se3, true);
    }
    /**
    * \brief generates se3 from VICON world2camera file
    * \param file_name path to world2camera file
    */
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

        // Convert left-handed VICON axis to right-handed Azure Kinect axis
        rotate_by_x(m_se3, transformation);
        
        reverse_x(m_se3, m_se3, true);
    }

    /**
    * \brief initializes charuco board
    * 
    * only boards with 5x7, 14x9, 4x3 are supported. 4x3 board is the default one.
    */
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
            throw std::runtime_error("Invalid board!");
            break;
        }
        m_params = aruco::DetectorParameters::create();
    }

    /**
    * \brief pipeline to get intrinsic and extrinsic calibrations
    */
    bool intrinsics_extrinsics(const char *file_path, Mat &cameraMatrix, Mat &distCoeffs, linmath::mat4x4 extrinsics_mat)
    {
        k4a_playback_t playback_handle = NULL;
        if (k4a_playback_open(file_path, &playback_handle) != K4A_RESULT_SUCCEEDED)
        {
            throw std::runtime_error("Failed to open recording!");
        }
        k4a_calibration_t calibration;
        k4a_result_t calibration_result = k4a_playback_get_calibration(playback_handle, &calibration);
        if (calibration_result != K4A_RESULT_SUCCEEDED)
        {
            k4a_playback_close(playback_handle);
            throw std::runtime_error("Failed to get calibration!");
        }
        get_intrinsics(calibration, cameraMatrix, distCoeffs);
        get_extrinsics(calibration, extrinsics_mat);

        k4a_playback_close(playback_handle);
        return true;
    }

    /**
    * \brief get extrinsic calibration of Azure Kinect device
    */
    void get_extrinsics(k4a_calibration_t calibration, linmath::mat4x4 extrinsics_mat)
    {
        k4a_calibration_extrinsics_t extrinsics =
            calibration.extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH];

        linmath::mat4x4_translate(extrinsics_mat,
                                  extrinsics.translation[0],
                                  extrinsics.translation[1],
                                  extrinsics.translation[2]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                extrinsics_mat[i][j] = extrinsics.rotation[3 * j + i]; // linmath uses column index first
    }

    /**
    * \brief get intrinsic calibration from Azure Kinect device as charuco needs
    */
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

    /**
    * \brief detect charuco pose
    * \param input_video_path path to the video file that contains charuco board
    * 
    * only the first frame is used. m_rvec and m_tvec are obtained here.
    */
    void detect_pose(string input_video_path)
    {
        VideoCapture inputVideo;
        inputVideo.open(input_video_path);

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

    /**
    * \brief generate se3 from m_rvec and m_tvec
    */
    void se3_from_rvec_tvec()
    {
        Mat rotation_cv;
        Rodrigues(m_rvec, rotation_cv);
        linmath::mat4x4 transformation;

        linmath::mat4x4_translate(transformation, (float)m_tvec(0), (float)m_tvec(1), (float)m_tvec(2));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                transformation[i][j] = (float)rotation_cv.at<double>(j, i); // linmath uses column index first

        linmath::mat4x4_dup(m_se3, transformation);

        reverse_x(m_se3, m_se3, false);
    }

    Ptr<aruco::Dictionary> m_dictionary;
    Ptr<aruco::CharucoBoard> m_board;
    Ptr<aruco::DetectorParameters> m_params;
    int m_board_type = 4; // Change board_type according to your charuco board
    Mat m_cameraMatrix, m_distCoeffs;
    Vec3d m_rvec, m_tvec;
    linmath::mat4x4 m_se3, m_se3_inverse;
    linmath::mat4x4 m_extrinsics;
};

} // namespace k4aviewer

#endif
