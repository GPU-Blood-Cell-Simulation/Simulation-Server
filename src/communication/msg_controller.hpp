#pragma once

#include <string>
#include <set>

#include "msg_receiver.hpp"
#include "../graphics/camera.hpp"


class MsgController {
public:
    MsgController(int server_port, const std::string& server_address = "localhost");
    ~MsgController();

    void setCamera(graphics::Camera* camera);
    void handleMsgs();

private:
    MsgReceiver receiver;
    graphics::Camera* camera = nullptr;

    std::set<Event> activeEvents;

    void adjustParameters();
};
