#include "msg_controller.hpp"

#include <stdexcept>
#include <iostream>

#include <glad/glad.h>

#include "../config/graphics.hpp"


MsgController::MsgController(int server_port, const std::string &server_address):
    communicationEndpoint(server_port, server_address)
{
}


void MsgController::setCamera(graphics::Camera *camera)
{
    this->camera = camera;
}

void MsgController::setStreamEndCallback(const std::function<void(void)> &callback)
{
    streamEndCallback = callback;
}

void MsgController::handleMsgs()
{
    Event received_event = communicationEndpoint.pollEvents();

    if (camera == nullptr)
        return;

    switch (received_event.state)
    {
    case EventState::start:
        activeEvents.insert(received_event.eventType);
        break;

    case EventState::stop:
        activeEvents.erase(received_event.eventType);
    
    case EventState::notRelevant:
        handleSingleMsgs(received_event);
        break;

    default:
        throw std::runtime_error("Unknown event state");
    }

    adjustParameters();
}


void MsgController::successfulStreamEndInform()
{
    Event event(EventType::streamSuccessfullyEnded);
    communicationEndpoint.SendEvent(event);
}


void MsgController::adjustParameters()
{
    for (auto& event : activeEvents) {
        switch (event)
        {
        case EventType::cameraLeft:
            camera->moveLeft();
            break;
        
        case EventType::cameraRight:
            camera->moveRight();
            break;

        case EventType::cameraForward:
            camera->moveForward();
            break;

        case EventType::cameraBack:
            camera->moveBack();
            break;

        case EventType::cameraAscend:
            camera->ascend();
            break;

        case EventType::cameraDescend:
            camera->descend();
            break;

        case EventType::cameraRotateLeft:
            camera->rotateLeft();
            break;

        case EventType::cameraRotateRight:
            camera->rotateRight();
            break;

        case EventType::cameraRotateUp:
            camera->rotateUp();
            break;

        case EventType::cameraRotateDown:
            camera->rotateDown();
            break;

        default:
            throw std::runtime_error("Unexpected event type");
        }
    }
}


void MsgController::handleSingleMsgs(Event event)
{
        switch (event.eventType)
        {
        case EventType::newConnection:
            std::cout << "New Connection established\n";
            break;
        
        case EventType::peerDisconnected:
            activeEvents.clear();
            std::cout << "Peer disconnected\n";
            break;

        case EventType::togglePolygonMode:
            glPolygonMode(GL_FRONT_AND_BACK, (VEIN_POLYGON_MODE = (VEIN_POLYGON_MODE == GL_LINE ? GL_FILL : GL_LINE)));
            break;

        case EventType::toggleSpringsRendering:
            BLOOD_CELL_SPRINGS_RENDER = !BLOOD_CELL_SPRINGS_RENDER;
            break;

        case EventType::stopRendering:
            std::cout << "Client requested simulation abort\n";
            if (streamEndCallback) {
                streamEndCallback();
            }
            break;

        default:
            // Other types of messages are ignored
            break;
        }
}
