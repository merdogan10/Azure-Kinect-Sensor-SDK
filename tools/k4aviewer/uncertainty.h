// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef UNCERTAINTY_H
#define UNCERTAINTY_H

// System headers
//
#include <fstream>

// Library headers
//

// Project headers
//
#include "charuco.h"
#include "utils.h"
using namespace cv;
using namespace std;
namespace k4aviewer
{
class Uncertainty
{
public:
    Uncertainty(string input_video, string depth_folder) : m_video(input_video, 0), m_charuco(input_video)
    {
        ofstream cv_out(depth_folder + "\\cv.txt");
        ofstream depth_out(depth_folder + "\\depth.txt");
        ofstream frames_out(depth_folder + "\\frames.txt");

        int number_of_frames = 440;
        vector<vector<Point3f>> corners_cv, corners_k4a_depth;
        vector<int> valid_frames;
        for (int i = 0; i < number_of_frames; i++)
        {
            bool result = false;
            result = m_video.next_capture_image();
            if (!result)
                continue;

            m_charuco.detect_charuco_pose(m_video.m_colorMat);
            if (!m_charuco.m_valid)
                continue;

            m_charuco.calculate_corners();
            //draw_charuco();

            cv_out << "[";
            depth_out << "[";

            for (int j = 0; j < m_charuco.m_calculated_corners.size(); j++)
            {
                // Only get 0, 2, 3 corners
                if (j == 1 || j > 3)
                    continue;
                k4a_float2_t point_2d;
                k4a_float3_t point_3d_dep;
                int valid;

                // 2d Charuco corner point
                point_2d.xy.x = static_cast<float>(m_charuco.m_calculated_corners[j].x);
                point_2d.xy.y = static_cast<float>(m_charuco.m_calculated_corners[j].y);

                k4a_float2_t point_2d_depth;
                k4a_calibration_color_2d_to_depth_2d(&m_video.m_calibration, &point_2d, m_video.m_depth_image, &point_2d_depth, &valid);
                int idx_col = (int)round(point_2d_depth.xy.x);
                int idx_row = (int)round(point_2d_depth.xy.y);

                float point_2d_depth_value = m_video.m_depthMat.at<uint16_t>(idx_row, idx_col);

                // Project 2d point to 3d point
                k4a_calibration_2d_to_3d(&m_video.m_calibration,
                                         &point_2d,
                                         point_2d_depth_value,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         K4A_CALIBRATION_TYPE_DEPTH,
                                         &point_3d_dep,
                                         &valid);
                Point3f dep_p(point_3d_dep.xyz.x, point_3d_dep.xyz.y, point_3d_dep.xyz.z);


                cv_out << m_charuco.m_calculated_corners_3d[j] * 1000;
                depth_out << dep_p;
                if (j != 3)
                {
                    cv_out << ",";
                    depth_out << ",";
                }
            }

            cv_out << "]" << endl;
            depth_out << "]" << endl;
            frames_out << i << endl;
            m_video.release_images();
        }
        m_video.close_playback();
    }

    void draw_charuco() {
        Mat color_copy;
        m_video.m_colorMat.copyTo(color_copy);
        aruco::drawDetectedCornersCharuco(color_copy,
                                          m_charuco.m_detected_corners,
                                          m_charuco.m_detected_ids,
                                          Scalar(0, 0, 255));
        aruco::drawDetectedCornersCharuco(color_copy,
                                          m_charuco.m_outer_corners,
                                          m_charuco.m_outer_ids,
                                          Scalar(0, 255, 0));

        aruco::drawAxis(color_copy,
                        m_charuco.m_cameraMatrix,
                        m_charuco.m_distCoeffs,
                        m_charuco.m_rvec,
                        m_charuco.m_tvec,
                        0.1f);
       Mat croppedFrame = color_copy(Rect(int(m_charuco.m_outer_corners[3].x) - 300,
                                           int(m_charuco.m_outer_corners[3].y) - 300,
                                           int(m_charuco.m_outer_corners[2].x - m_charuco.m_outer_corners[3].x) + 600,
                                           int(m_charuco.m_outer_corners[0].y - m_charuco.m_outer_corners[3].y) + 600));
        show_image("Projected", croppedFrame, 0, 0, true);
        //show_image("Projected", color_copy, 0, 0, true);
    }

    Charuco m_charuco;

    Video m_video;
};
} // namespace k4aviewer

#endif
