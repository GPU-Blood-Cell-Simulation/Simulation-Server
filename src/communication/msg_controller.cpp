#include "msg_controller.hpp"

#include <stdexcept>

#include <iostream>


MsgController::MsgController(int server_port, const std::string &server_address):
    receiver(server_port, server_address)
{
}

MsgController::~MsgController()
{
}

void MsgController::setCamera(graphics::Camera *camera)
{
    this->camera = camera;
}

void MsgController::handleMsgs()
{
    Event received_event = receiver.pollEvents();

    if (camera == nullptr)
        return;

    switch (received_event.state)
    {
    case EventState::start:
        activeEvents.insert(received_event.event);
        break;

    case EventState::stop:
        activeEvents.erase(received_event.event);
    
    case EventState::notRelevant:
        switch (received_event.event)
        {
        case EventType::newConnection:
            std::cout << "New Connection established\n";
            break;
        
        case EventType::peerDisconnected:
            activeEvents.clear();
            std::cout << "Peer disconnected\n";
            break;

        default:
            break;
        }
        break;

    default:
        throw std::runtime_error("Unknown event state");
    }

    /* TODO: handle disconnection */

    adjustParameters();
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
