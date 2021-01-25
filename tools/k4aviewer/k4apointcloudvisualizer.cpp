// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Associated header
//
#include "k4apointcloudvisualizer.h"

// System headers
//

// Library headers
//
#include <ratio>

// Project headers
//
#include "k4acolorimageconverter.h"
#include "k4adepthpixelcolorizer.h"
#include "k4astaticimageproperties.h"
#include "k4aviewerutil.h"
#include "perfcounter.h"

using namespace k4aviewer;

namespace
{
// Background color of point cloud viewer - dark grey
//
const ImVec4 ClearColor = { 0.05f, 0.05f, 0.05f, 0.0f };

// Resolution of the point cloud texture
//
constexpr ImageDimensions PointCloudVisualizerTextureDimensions = { 1280, 1152 };

} // namespace

GLenum K4APointCloudVisualizer::InitializeTexture(std::shared_ptr<K4AViewerImage> *texture) const
{
    return K4AViewerImage::Create(texture, nullptr, m_dimensions, GL_RGBA);
}

PointCloudVisualizationResult K4APointCloudVisualizer::UpdateTexture(std::shared_ptr<K4AViewerImage> *texture,
                                                                     const k4a::capture &capture,
                                                                     const k4a::capture &capture2)
{
    // Set up rendering to a texture
    //
    glBindRenderbuffer(GL_RENDERBUFFER, m_depthBuffer.Id());
    glBindFramebuffer(GL_FRAMEBUFFER, m_frameBuffer.Id());
    CleanupGuard frameBufferBindingGuard([]() { glBindFramebuffer(GL_FRAMEBUFFER, 0); });

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthBuffer.Id());

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, static_cast<GLuint>(**texture), 0);
    const GLenum drawBuffers = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &drawBuffers);

    const GLenum frameBufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (frameBufferStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        return PointCloudVisualizationResult::OpenGlError;
    }

    glViewport(0, 0, m_dimensions.Width, m_dimensions.Height);

    glEnable(GL_DEPTH_TEST);
    glClearColor(ClearColor.x, ClearColor.y, ClearColor.z, ClearColor.w);
    glClearDepth(1.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const linmath::vec2 displayDimensions{ static_cast<float>(m_dimensions.Width),
                                           static_cast<float>(m_dimensions.Height) };
    m_viewControl.GetPerspectiveMatrix(m_projection, displayDimensions);
    m_viewControl.GetViewMatrix(m_view);

    linmath::mat4x4 view_to_show;
    linmath::mat4x4_mul(view_to_show, m_view, m_se3);

    m_pointCloudRenderer.UpdateViewProjection(view_to_show, m_projection);

    // Update the point cloud renderer with the latest point data
    //
    PointCloudVisualizationResult result = UpdatePointClouds(capture,
                                                             m_pointCloudRenderer,
                                                             m_transformation,
                                                             m_transformedDepthImage,
                                                             m_xyzTexture,
                                                             m_lastCapture,
                                                             m_pointCloudColorization,
                                                             m_pointCloudConverter);
    if (result != PointCloudVisualizationResult::Success)
    {
        return result;
    }

    GLenum renderStatus = m_pointCloudRenderer.Render();

    linmath::mat4x4 view_to_show2;
    linmath::mat4x4_mul(view_to_show2, m_view, m_se3_2);

    //linmath::mat4x4_translate_in_place(view_to_show2, (float)-1.05, (float)0.05, 0); //shader fit
    //linmath::mat4x4_translate_in_place(view_to_show2, (float)-1.08, 0, 0); // color fit

    m_pointCloudRenderer2.UpdateViewProjection(view_to_show2, m_projection);

    PointCloudVisualizationResult result2 = UpdatePointClouds(capture2,
                                                             m_pointCloudRenderer2,
                                                             m_transformation2,
                                                             m_transformedDepthImage2,
                                                             m_xyzTexture2,
                                                             m_lastCapture2,
                                                             m_pointCloudColorization2,
                                                             m_pointCloudConverter2);
    if (result2 != PointCloudVisualizationResult::Success)
    {
        return result2;
    }
    GLenum renderStatus2 = m_pointCloudRenderer2.Render();

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    if (renderStatus != GL_NO_ERROR || renderStatus2 != GL_NO_ERROR)
    {
        return PointCloudVisualizationResult::OpenGlError;
    }

    return PointCloudVisualizationResult::Success;
}

void K4APointCloudVisualizer::ProcessMouseMovement(const linmath::vec2 displayDimensions,
                                                   const linmath::vec2 mousePos,
                                                   const linmath::vec2 mouseDelta,
                                                   MouseMovementType movementType)
{
    m_viewControl.ProcessMouseMovement(displayDimensions, mousePos, mouseDelta, movementType);
}

void K4APointCloudVisualizer::ProcessMouseScroll(const float yoffset)
{
    m_viewControl.ProcessMouseScroll(yoffset);
}

void K4APointCloudVisualizer::ResetPosition()
{
    m_viewControl.ResetPosition();
}
PointCloudVisualizationResult K4APointCloudVisualizer::SetColorizationStrategyIn(
                                                   PointCloudRenderer &pointCloudRenderer,
                                                   k4a::transformation &transformation,
                                                   k4a::image &transformedDepthImage,
                                                   OpenGL::Texture &xyzTexture,
                                                   k4a::capture &lastCapture,
                                                   k4a::image &pointCloudColorization,
                                                   GpuDepthToPointCloudConverter &pointCloudConverter,
                                                   const k4a::calibration &calibrationData,
                                                   k4a::image &colorXyTable,
                                                   k4a::image &depthXyTable)
{
    pointCloudRenderer.EnableShading(m_colorizationStrategy == ColorizationStrategy::Shaded);

    GLenum xyTableStatus = GL_NO_ERROR;
    if (m_colorizationStrategy == ColorizationStrategy::Color)
    {
        transformedDepthImage = k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
                                                     calibrationData.color_camera_calibration.resolution_width,
                                                     calibrationData.color_camera_calibration.resolution_height,
                                                     calibrationData.color_camera_calibration.resolution_width *
                                                         static_cast<int>(sizeof(DepthPixel)));

        xyTableStatus = pointCloudConverter.SetActiveXyTable(colorXyTable);
    }
    else
    {

        pointCloudColorization = k4a::image::create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                      calibrationData.depth_camera_calibration.resolution_width,
                                                      calibrationData.depth_camera_calibration.resolution_height,
                                                      calibrationData.depth_camera_calibration.resolution_width *
                                                          static_cast<int>(sizeof(BgraPixel)));

        xyTableStatus = pointCloudConverter.SetActiveXyTable(depthXyTable);
    }

    if (xyTableStatus != GL_NO_ERROR)
    {
        return PointCloudVisualizationResult::OpenGlError;
    }

    // Reset our reserved XYZ point cloud texture so it'll get resized the next time we go to render
    //
    xyzTexture.Reset();

    // If we've had data, force-refresh color pixels uploaded to GPU.
    // This allows us to switch shading modes while paused.
    //
    if (lastCapture)
    {
        return UpdatePointClouds(lastCapture,
                                 pointCloudRenderer,
                                 transformation,
                                 transformedDepthImage,
                                 xyzTexture,
                                 lastCapture,
                                 pointCloudColorization,
                                 pointCloudConverter);
    }

    return PointCloudVisualizationResult::Success;
}

PointCloudVisualizationResult K4APointCloudVisualizer::SetColorizationStrategy(ColorizationStrategy strategy)
{
    if (strategy == ColorizationStrategy::Color && !m_enableColorPointCloud)
    {
        throw std::logic_error("Attempted to set unsupported point cloud mode!");
    }

    m_colorizationStrategy = strategy;

    SetColorizationStrategyIn(m_pointCloudRenderer,
                              m_transformation,
                              m_transformedDepthImage,
                              m_xyzTexture,
                              m_lastCapture,
                              m_pointCloudColorization,
                              m_pointCloudConverter,
                              m_calibrationData,
                              m_colorXyTable,
                              m_depthXyTable);

    return SetColorizationStrategyIn(m_pointCloudRenderer2,
                              m_transformation2,
                              m_transformedDepthImage2,
                              m_xyzTexture2,
                              m_lastCapture2,
                              m_pointCloudColorization2,
                              m_pointCloudConverter2,
                              m_calibrationData2,
                              m_colorXyTable2,
                              m_depthXyTable2);
}

void K4APointCloudVisualizer::SetPointSize(int size)
{
    m_pointCloudRenderer.SetPointSize(size);
    m_pointCloudRenderer2.SetPointSize(size);
}

K4APointCloudVisualizer::K4APointCloudVisualizer(const bool enableColorPointCloud,
                                                 const k4a::calibration &calibrationData,
                                                 const k4a::calibration &calibrationData2,
                                                 linmath::mat4x4 se3,
                                                 linmath::mat4x4 se3_2) :
    m_dimensions(PointCloudVisualizerTextureDimensions),
    m_enableColorPointCloud(enableColorPointCloud),
    m_calibrationData(calibrationData),
    m_calibrationData2(calibrationData2)
{
    linmath::mat4x4_dup(m_se3, se3);
    linmath::mat4x4_dup(m_se3_2, se3_2);
    m_expectedValueRange = GetDepthModeRange(m_calibrationData.depth_mode);
    m_transformation = k4a::transformation(m_calibrationData);
    m_transformation2 = k4a::transformation(m_calibrationData2);

    glBindRenderbuffer(GL_RENDERBUFFER, m_depthBuffer.Id());
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_dimensions.Width, m_dimensions.Height);

    linmath::mat4x4_identity(m_view);
    linmath::mat4x4_identity(m_projection);

    m_viewControl.ResetPosition();

    if (enableColorPointCloud)
    {
        m_colorXyTable = m_pointCloudConverter.GenerateXyTable(m_calibrationData, K4A_CALIBRATION_TYPE_COLOR);
        m_colorXyTable2 = m_pointCloudConverter2.GenerateXyTable(m_calibrationData2, K4A_CALIBRATION_TYPE_COLOR);
    }

    m_depthXyTable = m_pointCloudConverter.GenerateXyTable(m_calibrationData, K4A_CALIBRATION_TYPE_DEPTH);
    m_depthXyTable2 = m_pointCloudConverter2.GenerateXyTable(m_calibrationData2, K4A_CALIBRATION_TYPE_DEPTH);

    SetColorizationStrategy(m_colorizationStrategy);
}

PointCloudVisualizationResult K4APointCloudVisualizer::UpdatePointClouds(const k4a::capture &capture,
                                                                         PointCloudRenderer &pointCloudRenderer,
                                                                         k4a::transformation &transformation,
                                                                         k4a::image &transformedDepthImage,
                                                                         OpenGL::Texture &xyzTexture,
                                                                         k4a::capture &lastCapture,
                                                                         k4a::image &pointCloudColorization,
                                                                         GpuDepthToPointCloudConverter &pointCloudConverter)
{
    k4a::image depthImage = capture.get_depth_image();
    if (!depthImage)
    {
        // Capture doesn't have depth info; drop the capture
        //
        return PointCloudVisualizationResult::MissingDepthImage;
    }

    k4a::image colorImage = capture.get_color_image();

    if (m_enableColorPointCloud)
    {
        if (!colorImage)
        {
            // Capture doesn't have color info; drop the capture
            //
            return PointCloudVisualizationResult::MissingColorImage;
        }

        if (m_colorizationStrategy == ColorizationStrategy::Color)
        {
            try
            {
                transformation.depth_image_to_color_camera(depthImage, &transformedDepthImage);
                depthImage = transformedDepthImage;
            }
            catch (const k4a::error &)
            {
                return PointCloudVisualizationResult::DepthToColorTransformationFailed;
            }
        }
    }

    GLenum glResult = pointCloudConverter.Convert(depthImage, &xyzTexture);
    if (glResult != GL_NO_ERROR)
    {
        return PointCloudVisualizationResult::DepthToXyzTransformationFailed;
    }

    lastCapture = capture;

    if (m_colorizationStrategy == ColorizationStrategy::Color)
    {
        pointCloudColorization = std::move(colorImage);
    }
    else
    {
        DepthPixel *srcPixel = reinterpret_cast<DepthPixel *>(depthImage.get_buffer());
        BgraPixel *dstPixel = reinterpret_cast<BgraPixel *>(pointCloudColorization.get_buffer());
        const BgraPixel *endPixel = dstPixel + (depthImage.get_size() / sizeof(DepthPixel));

        while (dstPixel != endPixel)
        {
            *dstPixel = K4ADepthPixelColorizer::ColorizeBlueToRed(*srcPixel,
                                                                  m_expectedValueRange.first,
                                                                  m_expectedValueRange.second);

            ++dstPixel;
            ++srcPixel;
        }
    }

    GLenum updatePointCloudResult = pointCloudRenderer.UpdatePointClouds(pointCloudColorization, xyzTexture);
    if (updatePointCloudResult != GL_NO_ERROR)
    {
        return PointCloudVisualizationResult::OpenGlError;
    }

    return PointCloudVisualizationResult::Success;
}
