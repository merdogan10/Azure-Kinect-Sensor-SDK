// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CALIBRATION_H
#define CALIBRATION_H

// System headers
//
#include <fstream>

// Library headers
//
#include "linmath.h"

// Project headers
//
#include "charuco.h"
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
 * \brief reverse x-dimension on both side of the matrix to get into OpenGL perspective.
 */
void move_into_GL(linmath::mat4x4 result, linmath::mat4x4 input)
{
    linmath::mat4x4 transform_x, temp_input;
    linmath::mat4x4_dup(temp_input, input);
    linmath::mat4x4_identity(transform_x);
    transform_x[0][0] = -1;
    linmath::mat4x4_mul(temp_input, temp_input, transform_x);
    linmath::mat4x4_mul(result, transform_x, temp_input);
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
    void get_se3_color(linmath::mat4x4 se3)
    {
        linmath::mat4x4_dup(se3, m_se3_color);
    }
    void get_se3_color_inverse(linmath::mat4x4 se3_inverse)
    {
        linmath::mat4x4_dup(se3_inverse, m_se3_color_inverse);
    }
    void get_se3_depth(linmath::mat4x4 se3)
    {
        linmath::mat4x4_dup(se3, m_se3_depth);
    }
    void get_se3_depth_inverse(linmath::mat4x4 se3_inverse)
    {
        linmath::mat4x4_dup(se3_inverse, m_se3_depth_inverse);
    }
    /**
    * \brief Main calibration pipeline
    * \param input_video
    * \param world2camera_file path to world2camera calibration file
    * \param icp_file path to icp calibration file
    * \param calibration_type can be only element of enum class CalibrationType
    */
    Calibration(string input_video, string world2camera_file, string icp_file, CalibrationType calibration_type) :
        m_charuco(input_video)
    {
        switch (calibration_type)
        {
        case k4aviewer::CalibrationType::Quaternion:
            se3_from_quat(world2camera_file);
            break;

        case k4aviewer::CalibrationType::Charuco:
            m_charuco.detect_charuco_pose(input_video);
            se3_from_charuco();
            break;

        case k4aviewer::CalibrationType::ICP:
            se3_from_icp(icp_file);
            break;

        default:
            throw std::runtime_error("Selected an unsupported calibration type!");
            break;
        }

        linmath::mat4x4_invert(m_se3_color_inverse, m_se3_color);
        linmath::mat4x4_invert(m_se3_depth_inverse, m_se3_depth);
    }
    Calibration(){};
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
        linmath::mat4x4_dup(m_se3_depth, transformation);
        linmath::mat4x4_mul(m_se3_color, m_se3_depth, m_charuco.m_extrinsics);
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
        linmath::mat4x4_rotate_X(m_se3_depth, transformation, Radians(180));
        linmath::mat4x4_mul(m_se3_color, m_se3_depth, m_charuco.m_extrinsics);
    }

    /**
    * \brief generate se3 from m_rvec and m_tvec coming as Charuco pose
    */
    void se3_from_charuco()
    {
        Mat rotation_cv;
        Rodrigues(m_charuco.m_rvec, rotation_cv);
        linmath::mat4x4 transformation;

        linmath::mat4x4_translate(transformation,
                                  (float)m_charuco.m_tvec(0),
                                  (float)m_charuco.m_tvec(1),
                                  (float)m_charuco.m_tvec(2));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                transformation[i][j] = (float)rotation_cv.at<double>(j, i); // linmath uses column index first

        linmath::mat4x4_dup(m_se3_color, transformation);
        linmath::mat4x4_mul(m_se3_depth, m_charuco.m_extrinsics, m_se3_color);
    }

    Charuco m_charuco;
    linmath::mat4x4 m_se3_color, m_se3_color_inverse;
    linmath::mat4x4 m_se3_depth, m_se3_depth_inverse;
};

} // namespace k4aviewer

#endif
