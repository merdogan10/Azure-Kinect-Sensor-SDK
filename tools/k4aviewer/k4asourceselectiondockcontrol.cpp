// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Associated header
//
#include "k4asourceselectiondockcontrol.h"

// System headers
//
#include <sstream>

// Library headers
//
#include "k4aimgui_all.h"
#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

// Project headers
//
#include "filesystem17.h"
#include "k4aaudiomanager.h"
#include "k4aimguiextensions.h"
#include "k4aviewererrormanager.h"
#include "k4arecordingdockcontrol.h"
#include "k4aviewerutil.h"
#include "k4awindowmanager.h"
#include "calibration.h"
#include "projection.h"
#include "uncertainty.h"
#include "linmath.h"
using namespace k4aviewer;

K4ASourceSelectionDockControl::K4ASourceSelectionDockControl()
{
    RefreshDevices();
}

K4ADockControlStatus K4ASourceSelectionDockControl::Show()
{
    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_FirstUseEver);
    if (ImGui::TreeNode("Open Device"))
    {
        ImGuiExtensions::K4AComboBox("Device S/N",
                                     "(No available devices)",
                                     ImGuiComboFlags_None,
                                     m_connectedDevices,
                                     &m_selectedDevice);

        if (ImGui::Button("Refresh Devices"))
        {
            RefreshDevices();
        }

        ImGui::SameLine();

        const bool openAvailable = !m_connectedDevices.empty();
        {
            ImGuiExtensions::ButtonColorChanger colorChanger(ImGuiExtensions::ButtonColor::Green, openAvailable);
            if (ImGuiExtensions::K4AButton("Open Device", openAvailable))
            {
                OpenDevice();
            }
        }

        ImGui::TreePop();
    }

    ImGui::NewLine();
    ImGui::Separator();
    ImGui::NewLine();

    if (ImGui::TreeNode("Open Recording"))
    {
        if (m_filePicker.Show())
        {
            OpenRecording(m_filePicker.GetPath());
        }

        ImGui::TreePop();
    }

    return K4ADockControlStatus::Ok;
}

void K4ASourceSelectionDockControl::RefreshDevices()
{
    m_selectedDevice = -1;

    const uint32_t installedDevices = k4a_device_get_installed_count();

    m_connectedDevices.clear();

    for (uint32_t i = 0; i < installedDevices; i++)
    {
        try
        {
            k4a::device device = k4a::device::open(i);
            m_connectedDevices.emplace_back(std::make_pair(i, device.get_serialnum()));
        }
        catch (const k4a::error &)
        {
            // We can't have 2 handles to the same device, and we need to open a device handle to check
            // its serial number, so we expect devices we already have open to fail here.  Ignore those.
            //
            continue;
        }
    }

    if (!m_connectedDevices.empty())
    {
        m_selectedDevice = m_connectedDevices[0].first;
    }

    const int audioRefreshStatus = K4AAudioManager::Instance().RefreshDevices();
    if (audioRefreshStatus != SoundIoErrorNone)
    {
        std::stringstream errorBuilder;
        errorBuilder << "Failed to refresh audio devices: " << soundio_strerror(audioRefreshStatus) << "!" << std::endl
                     << "Attempting to open microphones may fail!";

        K4AViewerErrorManager::Instance().SetErrorStatus(errorBuilder.str());
    }
}

void K4ASourceSelectionDockControl::OpenDevice()
{
    try
    {
        if (m_selectedDevice < 0)
        {
            K4AViewerErrorManager::Instance().SetErrorStatus("No device selected!");
            return;
        }

        k4a::device device = k4a::device::open(static_cast<uint32_t>(m_selectedDevice));
        K4AWindowManager::Instance().PushLeftDockControl(std14::make_unique<K4ADeviceDockControl>(std::move(device)));
    }
    catch (const k4a::error &e)
    {
        K4AViewerErrorManager::Instance().SetErrorStatus(e.what());
    }
}

void K4ASourceSelectionDockControl::OpenRecording(const std17::filesystem::path &path)
{
    try
    {
        path;
        string v1 = "v1", v2 = "v2";
        string c1 = "cn03", c2 = "cn06";
        string used_version = v1;
        string charuco_folder = "charuco_rotate";
        CalibrationType calibration_type = CalibrationType::Charuco;


        string captures_folder = "C:\\Users\\Mustafa\\Desktop\\thesis\\captures\\" + used_version;
        // string uncertainty_video_path = captures_folder + "\\board_depth\\cn06\\k4a_record.mkv";
        // Uncertainty certain(uncertainty_video_path);

        string calib_files[2][5];
        calib_files[0][(int)CalibrationType::Bundle] = captures_folder + "\\vicon_calibration\\" + c1 + "\\world2camera.txt";
        calib_files[0][(int)CalibrationType::Charuco] = captures_folder + "\\" + charuco_folder + "\\" + c1 + "\\k4a_record.mkv";
        calib_files[0][(int)CalibrationType::ICP] = captures_folder + "\\icp\\" + c1 + "\\icp.txt";
        calib_files[0][(int)CalibrationType::ColorICP] = captures_folder + "\\color_icp\\" + c1 + "\\color_icp.json";
        calib_files[1][(int)CalibrationType::Bundle] = captures_folder + "\\vicon_calibration\\" + c2 + "\\world2camera.txt";
        calib_files[1][(int)CalibrationType::Charuco] = captures_folder + "\\" + charuco_folder + "\\" + c2 + "\\k4a_record.mkv";
        calib_files[1][(int)CalibrationType::ICP] = captures_folder + "\\icp\\" + c2 + "\\icp.txt";
        calib_files[1][(int)CalibrationType::ColorICP] = captures_folder + "\\color_icp\\" + c2 + "\\color_icp.json";

        string calib_type_names[5];
        calib_type_names[(int)CalibrationType::Bundle] = "bundle";
        calib_type_names[(int)CalibrationType::Charuco] = "charuco";
        calib_type_names[(int)CalibrationType::ICP] = "icp";
        calib_type_names[(int)CalibrationType::ColorICP] = "coloricp";
        
        string input_video_path = calib_files[0][(int)CalibrationType::Charuco];
        string input_video_path2 = calib_files[1][(int)CalibrationType::Charuco];

        Calibration calib(input_video_path, calib_files[0][(int)calibration_type], calibration_type);
        Calibration calib2(input_video_path2, calib_files[1][(int)calibration_type], calibration_type);

        linmath::mat4x4 se3, se3_2;
        linmath::mat4x4 c2c_depth, c2c_color;

        switch (calibration_type)
        {
        case k4aviewer::CalibrationType::Bundle:
        case k4aviewer::CalibrationType::ColorICP:
            // se3' * se3_2
            calib.get_se3_color_inverse(se3);
            calib2.get_se3_color(se3_2);
            linmath::mat4x4_mul(c2c_color, se3, se3_2);
            calib.get_se3_depth_inverse(se3);
            calib2.get_se3_depth(se3_2);
            linmath::mat4x4_mul(c2c_depth, se3, se3_2);
            break;

        case k4aviewer::CalibrationType::ICP:
        case k4aviewer::CalibrationType::Charuco:
            // se3 * se3_2'
            calib.get_se3_color(se3);
            calib2.get_se3_color_inverse(se3_2);
            linmath::mat4x4_mul(c2c_color, se3, se3_2);
            calib.get_se3_depth(se3);
            calib2.get_se3_depth_inverse(se3_2);
            linmath::mat4x4_mul(c2c_depth, se3, se3_2);
            break;
        }
        
        string test_video_path = captures_folder + "v1\\charuco_move\\cn03\\k4a_record.mkv";
        string test_video_path2 = captures_folder + "v1\\charuco_move\\cn06\\k4a_record.mkv";
        /*ProjectionMode projection_modes[] = { ProjectionMode::Outer_3D_raycast_homography,
                                              ProjectionMode::Inner_3D,
                                              ProjectionMode::Inner_2D_calculated_corners,
                                              ProjectionMode::Inner_2D_detected_corners };
        string mode_str[] = { "template_matching", "3d", "2d_calculated", "2d_detected" };
        for (int i = 0; i < 4; i++)
        {
            string projection_stats_path = captures_folder +"\\"+ calib_type_names[(int)calibration_type] + '-' + mode_str[i] + '-' + "stats.txt";
            Projection proj(input_video_path, input_video_path2, c2c_color, projection_modes[i], projection_stats_path);
        }*/
        Projection proj(input_video_path, input_video_path2, c2c_color, ProjectionMode::Outer_3D_raycast_homography, captures_folder+"stats.txt");
        

        // Move c2c into OpenGL perspective
        move_into_GL(c2c_color, c2c_color);
        move_into_GL(c2c_depth, c2c_depth);

        k4a::playback recording = k4a::playback::open(input_video_path.c_str());
        k4a::playback recording2 = k4a::playback::open(input_video_path2.c_str());
        K4AWindowManager::Instance().PushLeftDockControl(std14::make_unique<K4ARecordingDockControl>(
            input_video_path, std::move(recording), std::move(recording2), c2c_depth, c2c_color));
    }
    catch (const k4a::error &e)
    {
        K4AViewerErrorManager::Instance().SetErrorStatus(e.what());
    }
}
