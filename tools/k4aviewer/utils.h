// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef UTILS_H
#define UTILS_H

// System headers
//

// Library headers
//
#include <opencv2/imgproc.hpp>
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
* \brief converts opencv 3d point to be used in projectPoints
*/
vector<Point3f> prepare_for_2d_reprojection(Vec3d &rvec, vector<Point3f> original)
{
    vector<Point3f> result;
    for (int i = 0; i < original.size(); i++)
    {
        Vec3d tvec(original[i].x, original[i].y, original[i].z);
        Mat R;
        Rodrigues(rvec, R);
        R = R.t();
        Mat new_tvec = -R * tvec;
        double *tmp = new_tvec.ptr<double>(0);
        result.push_back(Point3f((float)tmp[0] * 1000, (float)tmp[1] * 1000, (float)tmp[2] * 1000));
    }
    return result;
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
 * \brief get extrinsic calibration of Azure Kinect device in meters
 */
void get_extrinsics(k4a_calibration_t calibration, linmath::mat4x4 extrinsics_mat)
{
    k4a_calibration_extrinsics_t extrinsics =
        calibration.extrinsics[K4A_CALIBRATION_TYPE_COLOR][K4A_CALIBRATION_TYPE_DEPTH];
    float units_per_meter = 1000.0f;
    linmath::mat4x4_translate(extrinsics_mat,
                              extrinsics.translation[0] / units_per_meter,
                              extrinsics.translation[1] / units_per_meter,
                              extrinsics.translation[2] / units_per_meter);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            extrinsics_mat[i][j] = extrinsics.rotation[3 * j + i]; // linmath uses column index first
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

class Video
{
public:
    Video(string input_video, int seek = 8000) : m_input_video(input_video)
    {
        open_playback(m_input_video, m_playback_handle, m_calibration, seek);
    }
    Video(){};
    ~Video() = default;

    void release_images() {
        k4a_image_release(m_depth_image);
        k4a_image_release(m_color_image);
    }

    void close_playback() {
        k4a_playback_close(m_playback_handle);
    }

    /**
     * \brief get next captured depth and color images as both opencv matrix and k4a image
     */
    bool next_capture_image() {
        return next_capture_image(m_playback_handle, m_capture, m_depthMat, m_colorMat, m_depth_image, m_color_image);
    }

    string m_input_video;
    k4a_playback_t m_playback_handle;
    k4a_calibration_t m_calibration;
    k4a_capture_t m_capture;

    k4a_image_t m_depth_image, m_color_image;
    Mat m_depthMat, m_colorMat;

private:

    /**
    * \brief open playback and get calibration
    */
    void open_playback(string video, k4a_playback_t &playback_handle, k4a_calibration_t &calibration, int seek = 8000)
    {
        if (k4a_playback_open(video.c_str(), &playback_handle) != K4A_RESULT_SUCCEEDED)
        {
            throw std::runtime_error("Failed to open recording!");
        }
        k4a_result_t calibration_result = k4a_playback_get_calibration(playback_handle, &calibration);
        if (calibration_result != K4A_RESULT_SUCCEEDED)
        {
            k4a_playback_close(playback_handle);
            throw std::runtime_error("Failed to get calibration!");
        }
        k4a_playback_set_color_conversion(playback_handle, K4A_IMAGE_FORMAT_COLOR_BGRA32);
        // seek to 8th second
        k4a_playback_seek_timestamp(playback_handle, seek * 1000, K4A_PLAYBACK_SEEK_BEGIN);
    }

    /**
     * \brief get image with given image type from the capture
     *
     * Only one type of image can be get for one call
     */
    Mat get_image(k4a_capture_t &capture, k4a_image_t &capture_image, string capture_type)
    {
        int cv_image_type;
        if (capture_type == "color")
        {
            capture_image = k4a_capture_get_color_image(capture);
            cv_image_type = CV_8UC4;
        }
        else
        {
            capture_image = k4a_capture_get_depth_image(capture);
            cv_image_type = CV_16U;
        }
        if (capture_image == NULL)
        {
            throw std::runtime_error("Failed to capture image!");
        }
        int height = k4a_image_get_height_pixels(capture_image);
        int width = k4a_image_get_width_pixels(capture_image);
        uint8_t *imgData = k4a_image_get_buffer(capture_image);

        Mat cv_capture_image(height, width, cv_image_type, (void *)imgData, Mat::AUTO_STEP);
        if (capture_type == "color")
            cvtColor(cv_capture_image, cv_capture_image, COLOR_BGRA2BGR);

        return cv_capture_image;
    }

    /**
     * \brief get next captured depth and color images as both opencv matrix and k4a image
     */
    bool next_capture_image(k4a_playback_t &playback_handle,
                            k4a_capture_t &capture,
                            Mat &depthMat,
                            Mat &colorMat,
                            k4a_image_t &depth_image,
                            k4a_image_t &color_image)
    {
        k4a_stream_result_t stream_result;
        stream_result = k4a_playback_get_next_capture(playback_handle, &capture);
        if (stream_result != K4A_STREAM_RESULT_SUCCEEDED || capture == NULL)
        {
            throw std::runtime_error("Failed to get capture!");
        }
        try
        {
            depthMat = get_image(capture, depth_image, "depth");
            colorMat = get_image(capture, color_image, "color");
        }
        catch (const std::exception &)
        {
            return false;
        }
        return true;
    }
};

} // namespace k4aviewer


#endif
