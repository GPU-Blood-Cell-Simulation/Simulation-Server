#pragma once

#include <functional>
#include <string>
#include <set>

#include "server_endpoint.hpp"
#include "../graphics/camera.hpp"


/// <summary>
/// Controller for message in visualization streaming
/// </summary>
class MsgController {
public:
    MsgController(int server_port, const std::string& server_address = "localhost");

    /// <summary>
	/// Sets camera, which is needed to react to client messages
	/// </summary>
	/// <param name="camera">Pointer to the camera used during rendering</param>
    void setCamera(graphics::Camera* camera);

    void setStreamEndCallback(const std::function<void()>& callback);

    /// <summary>
    /// Detects message type and execute specific action
    /// </summary>
    void handleMsgs();

    void successfulStreamEndInform();

private:
    ServerCommunicationEndpoint communicationEndpoint;
    graphics::Camera* camera = nullptr;

    std::set<EventType> activeEvents;

    std::function<void()> streamEndCallback;

    void adjustParameters();
    void handleSingleMsgs(Event event);
};
