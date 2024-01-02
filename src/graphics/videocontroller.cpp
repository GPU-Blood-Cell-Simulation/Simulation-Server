#include "videocontroller.hpp"

#include "../config/graphics.hpp"
#include <stdexcept>
#include <iostream>


int VideoController::numberOfInstances = 0;


static void bus_error_handler(GstBus *bus, GstMessage *msg, gpointer data)
{
	GError *err;
	gchar *debugTmp;
	std::string errorSrc(GST_OBJECT_NAME(msg->src));
	
	gst_message_parse_error (msg, &err, &debugTmp);

	std::string debugInfo(debugTmp);
	std::string errorMsg(err->message);

	g_error_free (err);
	g_free (debugTmp);

	throw std::runtime_error(
		"Streaming pipeline error."
		" Source: " + errorSrc +
		" Error message: " + errorSrc + 
		" Debug info: " + debugInfo
	);
}


VideoController::VideoController():
	pixels(windowHeight * windowWidth * 3), timestamp(0)
{
	if (++numberOfInstances == 1) {
		/* Init GStreamer */
		gst_init(NULL, NULL);
	}

	/* Create pipeline elements */
	appsrc = createPipelineElement("appsrc", "source");
	convert = createPipelineElement("videoconvert", "converter");
	tee = createPipelineElement("tee", "tee");
	
	pipeline = gst_pipeline_new("pipeline");
	if (!pipeline) {
		throw std::runtime_error("Error while creating streaming pipeline");
	}

	/* Create and setup pipeline bus */
	bus = gst_element_get_bus(pipeline);
	if (!bus) {
		gst_object_unref(pipeline);
		throw std::runtime_error("Cannot get a pipeline bus");
	}

	/* Add pipeline error callback */
	gst_bus_add_signal_watch(bus);
  	g_signal_connect(bus, "message::error", G_CALLBACK(bus_error_handler), NULL);

	gst_bin_add_many(GST_BIN(pipeline), appsrc, convert, tee, NULL);

	/* Appsrc setup */
  	g_object_set(G_OBJECT(appsrc), "caps",
  		gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "RGB",
			"width", G_TYPE_INT, windowWidth,
			"height", G_TYPE_INT, windowHeight,
			"framerate", GST_TYPE_FRACTION, 0, 1,
			NULL), NULL);

	g_object_set(G_OBJECT(appsrc),
		"stream-type", 0, // GST_APP_STREAM_TYPE_STREAM
		"format", GST_FORMAT_TIME,
		"is-live", TRUE,
		"block", TRUE,
		NULL);

	/* Link elements */
	if (gst_element_link_many(appsrc, convert, tee, NULL) != TRUE)
	{
		gst_object_unref(bus);
		gst_object_unref(pipeline);
		throw std::runtime_error("Cannot link appsrc to tee");
	}
}


GstElement *VideoController::createPipelineElement(const std::string& factoryName, const std::string& name)
{
    GstElement *result = gst_element_factory_make(factoryName.c_str(), name.c_str());
	if (!result) {
		throw std::runtime_error("Error while creating " + name + " element");
	}

	return result;
}


VideoController::~VideoController()
{
	if (teeStreamPad != NULL) {
		gst_element_release_request_pad(tee, teeStreamPad);
		gst_object_unref(teeStreamPad);
	}

	if (teeFilePad != NULL) {
		gst_element_release_request_pad(tee, teeFilePad);
		gst_object_unref(teeFilePad);
	}

	gst_object_unref(bus);
	gst_element_set_state (pipeline, GST_STATE_NULL);
	gst_object_unref (GST_OBJECT(pipeline));

	if (--numberOfInstances == 0) {
		gst_deinit();
	}
}

void VideoController::SetUpStreaming(int port, const std::string &host)
{
	queueStream = createPipelineElement("queue", "streamQueue");
	x264encStream = createPipelineElement("x264enc", "streamEncoder");
	rtph264pay = createPipelineElement("rtph264pay", "payload");
	udpsink = createPipelineElement("udpsink", "sink");

	g_object_set(G_OBJECT(udpsink),
		"host", host.data(),
		"port", port,
		NULL);

	// Set the fastest preset
	g_object_set(G_OBJECT(x264encStream), "speed-preset", 1, NULL);

	// Tune for minimal latency
	g_object_set(G_OBJECT(x264encStream), "tune", 0x00000004, NULL);

	// Tune for animation
	g_object_set(G_OBJECT(x264encStream), "psy-tune", 2, NULL);

	gst_bin_add_many(GST_BIN(pipeline), queueStream, x264encStream, rtph264pay, udpsink, NULL);

	if (gst_element_link_many(queueStream, x264encStream, rtph264pay, udpsink, NULL) != TRUE) {
		throw std::runtime_error("Error while linking streaming elements");
	}

	/* Manually link streaming pipeline with tee element */
	teeStreamPad = gst_element_request_pad_simple(tee, "src_%u");
	GstPad *streamingPad = gst_element_get_static_pad(queueStream, "sink");

	GstPadLinkReturn ret = gst_pad_link(teeStreamPad, streamingPad);
	gst_object_unref(streamingPad);

	if (ret != GST_PAD_LINK_OK) {
		throw std::runtime_error("Error while linking stream to tee");
	}
}


void VideoController::SetUpRecording(const std::string &file_name)
{
	GstPadLinkReturn ret;

	queueFile = createPipelineElement("queue", "fileQueue");
	x264encFile = createPipelineElement("x264enc", "fileEncoder");
	muxer = createPipelineElement("matroskamux", "matroskamux");
	filesink = createPipelineElement("filesink", "filesink");
	h264parse = createPipelineElement("h264parse", "h264parse");

	g_object_set(G_OBJECT(filesink), "location", file_name.c_str(), NULL);

	gst_bin_add_many(GST_BIN(pipeline), queueFile, x264encFile, h264parse, muxer, filesink, NULL);

	if (gst_element_link_many(queueFile, x264encFile, h264parse, NULL) != TRUE ||
		gst_element_link_many(muxer, filesink, NULL) != TRUE) {
		throw std::runtime_error("Error while linking file elements");
	}

	/* Manually link file pipeline with tee element */
	teeFilePad = gst_element_request_pad_simple(tee, "src_%u");
	GstPad *filePad = gst_element_get_static_pad(queueFile, "sink");

	ret = gst_pad_link(teeFilePad, filePad);
	//gst_object_unref(filePad);
	
	if (ret != GST_PAD_LINK_OK) {
		throw std::runtime_error("Error while linking stream to tee");
	}

	/* Manually link muxer with filesink */
	mp4muxRequestPad = gst_element_request_pad_simple(muxer, "video_%u");
	GstPad *x264parsePad = gst_element_get_static_pad(h264parse, "src");

	ret = gst_pad_link(x264parsePad, mp4muxRequestPad);
	//gst_object_unref(x264parsePad);

	if (ret != GST_PAD_LINK_OK) {
		throw std::runtime_error("Error while linking parse to mux");
	}
}


void VideoController::StartPlayback()
{
	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);

	if (ret == GST_STATE_CHANGE_FAILURE) {
        throw std::runtime_error("Cannot set pipeline to playing state");
    }
}


void VideoController::Pause()
{
	GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PAUSED);

	if (ret == GST_STATE_CHANGE_FAILURE) {
        throw std::runtime_error("Cannot set pipeline to playing state");
    }
}


void VideoController::SendFrame()
{
	GstBuffer *buffer;

	glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

	buffer = gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_LAST, (gpointer)(pixels.data()), pixels.size(), 0, pixels.size(), NULL, NULL );

	GST_BUFFER_PTS(buffer) = timestamp;
	GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, 30);

	timestamp += GST_BUFFER_DURATION(buffer);

	if (gst_app_src_push_buffer((GstAppSrc*)appsrc, buffer) != GST_FLOW_OK) {
		throw std::runtime_error("Error while pushing data to buffer");
	}

	g_main_context_iteration(g_main_context_default(),FALSE);
}