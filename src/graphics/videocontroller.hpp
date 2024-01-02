#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <vector>
#include <glad/glad.h>
#include <string>


class VideoController
{
public:
    VideoController();
    VideoController(const VideoController&) = delete;
    ~VideoController();

    void SetUpStreaming(int port, const std::string& host);
    void SetUpRecording(const std::string& file_name);

    void StartPlayback();
    void Pause();

    void SendFrame();

private:
    static int numberOfInstances;

    /* Default elements */
    GstElement *pipeline;
    GstElement *appsrc;
    GstElement *convert;
    GstElement *tee;

    GstBus *bus;

    /* Streaming elements */
    GstElement *queueStream;
    GstElement *x264encStream;
    GstElement *rtph264pay;
    GstElement *udpsink;

    GstPad *teeStreamPad = NULL;
    std::vector<GLubyte> pixels;
    GstClockTime timestamp;
    
    /* Recording elements */
    GstElement *queueFile;
    GstElement *x264encFile;
    GstElement *h264parse;
    GstElement *muxer;
    GstElement *filesink;

    GstPad *teeFilePad = NULL;
    GstPad *mp4muxRequestPad = NULL;

    static GstElement *createPipelineElement(const std::string& factoryName, const std::string& name);
};
