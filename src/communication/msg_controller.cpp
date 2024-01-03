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
    ReceivedEvent received_event = receiver.pollMessages();

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
        case Event::newConnection:
            std::cout << "New Connection established\n";
            break;
        
        case Event::peerDisconnected:
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
        case Event::cameraLeft:
            camera->moveLeft();
            break;
        
        case Event::cameraRight:
            camera->moveRight();
            break;

        case Event::cameraForward:
            camera->moveForward();
            break;

        case Event::cameraBack:
            camera->moveBack();
            break;

        case Event::cameraAscend:
            camera->ascend();
            break;

        case Event::cameraDescend:
            camera->descend();
            break;

        case Event::cameraRotateLeft:
            camera->rotateLeft();
            break;

        case Event::cameraRotateRight:
            camera->rotateRight();
            break;

        case Event::cameraRotateUp:
            camera->rotateUp();
            break;

        case Event::cameraRotateDown:
            camera->rotateDown();
            break;

        default:
            throw std::runtime_error("Unexpected event type");
        }
    }
}
