// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CHARUCO_H
#define CHARUCO_H

// System headers
//

// Library headers
//
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "linmath.h"

// Project headers
//
#include "utils.h"
using namespace cv;
using namespace std;
namespace k4aviewer
{
class Charuco
{
public:

    Charuco(string input_video)
    {
        intrinsics_extrinsics(input_video.c_str(), m_cameraMatrix, m_distCoeffs, m_extrinsics);
        init_board(m_board_type, m_dictionary, m_board, m_params);
    }
    Charuco(){};
    ~Charuco() = default;

    /**
     * \brief initializes charuco board
     */
    void init_board() {
        init_board(m_board_type, m_dictionary, m_board, m_params);
    }

    /**
     * \brief Charuco Pose detector when only have image path
     */
    void detect_charuco_pose(string input_video_path) {
        VideoCapture inputVideo;
        inputVideo.open(input_video_path);

        inputVideo.grab();
        Mat image;
        inputVideo.retrieve(image);
        detect_charuco_pose(image);
    }

    /**
    * \brief Charuco Pose detector when only have image
    */
    void detect_charuco_pose(Mat image) {
        detect_charuco_pose(
            image, m_board, m_params, m_cameraMatrix, m_distCoeffs, m_detected_corners, m_detected_ids, m_rvec, m_tvec);
    }

    void calculate_corners()
    {
        calculate_charuco_corners(m_calculated_corners, m_calculated_corners_3d, m_calculated_ids, false);
        calculate_charuco_corners(m_outer_corners, m_outer_corners_3d, m_outer_ids, true);
    }
    
    bool m_valid;
    Vec3d m_rvec, m_tvec;
    vector<Point2f> m_detected_corners, m_calculated_corners, m_outer_corners;
    vector<Point3f> m_calculated_corners_3d, m_outer_corners_3d;
    vector<int> m_detected_ids, m_calculated_ids, m_outer_ids;
    Mat m_cameraMatrix, m_distCoeffs;
    linmath::mat4x4 m_extrinsics;

private:
    /**
     * \brief initializes charuco board
     *
     * only boards with 5x7, 14x9, 4x3 are supported. 4x3 board is the default one.
     */
    void init_board(int board_type,
                    Ptr<aruco::Dictionary> &dictionary,
                    Ptr<aruco::CharucoBoard> &board,
                    Ptr<aruco::DetectorParameters> &params)
    {
        switch (board_type)
        {
        case 5:
            dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
            board = aruco::CharucoBoard::create(5, 7, 0.04f, 0.02f, dictionary);
            break;
        case 14:
            dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_250);
            board = aruco::CharucoBoard::create(14, 9, 0.02f, 0.01556f, dictionary);
            break;
        case 4:
            dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);
            board = aruco::CharucoBoard::create(4, 3, 0.053f, 0.04f, dictionary);
            break;
        case 9:
            dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
            board = cv::aruco::CharucoBoard::create(4, 3, 0.09f, 0.0675f, dictionary);
            break;
        default:
            throw std::runtime_error("Invalid board!");
            break;
        }
        params = aruco::DetectorParameters::create();
    }

    /**
     * \brief standalone Charuco Pose Detector
     */
    void detect_charuco_pose(Mat image,
                             Ptr<aruco::CharucoBoard> &board,
                             Ptr<aruco::DetectorParameters> &params,
                             Mat &cameraMatrix,
                             Mat &distCoeffs,
                             vector<Point2f> &detected_corners,
                             vector<int> &detected_ids,
                             Vec3d &rvec,
                             Vec3d &tvec)
    {
        m_valid = false;
        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners;
        aruco::detectMarkers(image, board->dictionary, markerCorners, markerIds, params);
        // if at least one marker detected
        if (markerIds.size() > 0)
        {
            aruco::interpolateCornersCharuco(
                markerCorners, markerIds, image, board, detected_corners, detected_ids, cameraMatrix, distCoeffs);
            // if at least one corner detected
            if (detected_ids.size() > 0)
            {
                m_valid = aruco::estimatePoseCharucoBoard(
                    detected_corners, detected_ids, board, cameraMatrix, distCoeffs, rvec, tvec);
            }
        }
    }

    /**
     * \brief calculate corner points by getting Horizontal and Vertical chess square vectors
     */
    vector<Point3f> get_corners_in_camera_world(double side, Vec3d rvec, Vec3d tvec, bool board_corners = false)
    {
        // compute rot_mat
        Mat rot_mat;
        Rodrigues(rvec, rot_mat);

        // transpose of rot_mat for easy columns extraction
        Mat rot_mat_t = rot_mat.t();

        // the two E-O and F-O vectors
        double *tmp = rot_mat_t.ptr<double>(0);
        Point3f cam_world_E((float)(tmp[0] * side), (float)(tmp[1] * side), (float)(tmp[2] * side));

        tmp = rot_mat_t.ptr<double>(1);
        Point3f cam_world_F((float)(tmp[0] * side), (float)(tmp[1] * side), (float)(tmp[2] * side));

        // convert tvec to point
        Point3f tvec_3f((float)tvec[0], (float)tvec[1], (float)tvec[2]);
        vector<Point3f> result;
        if (board_corners)
        {
            // 4 corners of the board
            vector<Point3f> ret(4, tvec_3f);
            ret[1] += 4 * cam_world_E;
            ret[2] += 4 * cam_world_E + 3 * cam_world_F;
            ret[3] += 3 * cam_world_F;
            return ret;
        }
        else
        {
            // 6 inner corners of charuco
            vector<Point3f> ret(6, tvec_3f);
            ret[0] += cam_world_E + cam_world_F;
            ret[1] += 2 * cam_world_E + cam_world_F;
            ret[2] += 3 * cam_world_E + cam_world_F;
            ret[3] += cam_world_E + 2 * cam_world_F;
            ret[4] += 2 * cam_world_E + 2 * cam_world_F;
            ret[5] += 3 * cam_world_E + 2 * cam_world_F;
            return ret;
        }
    }

    /**
    * \brief calculate 3d and 2d corners by the 3d detected pose
    */
    void calculate_charuco_corners(vector<Point2f> &corners,
                                   vector<Point3f> &corners_3d,
                                   vector<int> &ids,
                                   bool is_board_corners = false)
    {
        corners_3d = get_corners_in_camera_world(m_board->getSquareLength(), m_rvec, m_tvec, is_board_corners);
        vector<Point3f> converted;
        converted = prepare_for_2d_reprojection(m_rvec, corners_3d);
        projectPoints(converted, m_rvec, m_tvec, m_cameraMatrix, m_distCoeffs, corners);

        ids.resize(corners_3d.size());
        generate(ids.begin(), ids.end(), [n = 0]() mutable { return n++; });
    }

    Ptr<aruco::Dictionary> m_dictionary;
    Ptr<aruco::CharucoBoard> m_board;
    Ptr<aruco::DetectorParameters> m_params;
    int m_board_type = 4; // Change board_type according to your charuco board
};

} // namespace k4aviewer

#endif
