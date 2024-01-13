#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <vector>
#include <glad/glad.h>
#include <string>

/// <summary>
/// Controls visualization's streaming and recording 
/// </summary>
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
    void EndPlayback();

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
    GstElement *h264encStream;
    GstElement *h264parserStream;
    GstElement *muxerStream;
    GstElement *tcpsink;

    GstPad *teeStreamPad = NULL;
    std::vector<GLubyte> pixels;
    GstClockTime timestamp;
    
    /* Recording elements */
    GstElement *queueFile;
    GstElement *videoFlip;
    GstElement *jpegencFile;
    GstElement *muxer;
    GstElement *filesink;

    GstPad *teeFilePad = NULL;
    GstPad *mp4muxRequestPad = NULL;

    static GstElement *createPipelineElement(const std::string& factoryName, const std::string& name);

    static void bus_error_handler(GstBus *bus, GstMessage *msg, gpointer data);
};
