#pragma once

#define ENET_IMPLEMENTATION
#include <enet/enet.h>

#include <cstdint>
#include <string>


typedef std::uint8_t eventType;

/// <summary>
/// Enum for distinct event types
/// </summary>
enum class Event: eventType {
    noMessage = 0,

    cameraLeft,
    cameraRight,
    cameraForward,
    cameraBack,
    cameraAscend,
    cameraDescend,

    cameraRotateLeft,
    cameraRotateRight,
    cameraRotateUp,
    cameraRotateDown,

    newConnection,
    peerDisconnected,

    invalidMessage,
};


/// <summary>
/// Enum for streaming states
/// </summary>
enum class EventState {
    start,
    stop,
    notRelevant
};

/// <summary>
/// Represents event received from client
/// </summary>
struct ReceivedEvent {
    Event event;
    EventState state;

    ReceivedEvent(Event event = Event::noMessage, EventState state = EventState::notRelevant);
};

/// <summary>
/// Represents message received from client
/// </summary>
class MsgReceiver {
public:
    MsgReceiver(int port, const std::string& address = "localhost");
    ~MsgReceiver();

    /// <summary>
	/// Checks for incoming events from the client. If no message was encountered, then
    /// It returns <c>ReceivedEvent<c> object with event set as <c>noMessage<c>. Function is not blocking.
	/// </summary>
	/// <returns>Event received from the client</returns>
    ReceivedEvent pollMessages() const;

private:
    ENetAddress address;
    ENetHost* server;

    ReceivedEvent parseMessage(const ENetEvent* event) const;
};
