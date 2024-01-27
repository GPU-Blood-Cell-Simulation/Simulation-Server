#pragma once

#define ENET_IMPLEMENTATION
#include <enet/enet.h>

#include <cstdint>
#include <string>


/// <summary>
/// Data structure of a single event
/// </summary>
typedef std::uint8_t event_t;


/// <summary>
/// Enum for distinct event types
/// </summary>
enum class EventType: event_t {
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

    togglePolygonMode,
    toggleSpringsRendering,
    toggleSpheresRendering,

    streamSuccessfullyEnded,
    stopRendering,

    newConnection,
    peerDisconnected,

    invalidMessage,
};


/// <summary>
/// Enum for streaming states
/// </summary>
enum class EventState: std::uint8_t {
    start,
    stop,
    notRelevant
};


/// <summary>
/// Represents event received from client
/// </summary>
struct Event {
    EventType eventType;
    EventState state;

    Event(EventType event = EventType::noMessage, EventState state = EventState::notRelevant);
};


/// <summary>
/// Communication handler. Receives and sends messages to client.
/// </summary>
class ServerCommunicationEndpoint {
public:
    ServerCommunicationEndpoint(int port, const std::string& address = "localhost");
    ~ServerCommunicationEndpoint();

    /// <summary>
	/// Checks for incoming events from the client. If no message was encountered, then
    /// returns <c>Event<c> object with event type set as <c>noMessage<c>. Function is not blocking.
	/// </summary>
	/// <returns>Event received from the client</returns>
    Event pollEvents();

    bool isConnected() const;
    void disconnect();

    void SendEvent(Event event) const;

private:
    /// <summary>
	/// Number of instances of this class
	/// </summary>
    static int instances;

    /// <summary>
	/// Server address
	/// </summary>
    ENetAddress address;

    /// <summary>
	/// Pointer to ENet server communication peer
	/// </summary>
    ENetHost* server;

    /// <summary>
	/// Pointer to ENet server communication peer
	/// </summary>
    ENetPeer* peer = NULL;

    static Event parseEvent(const ENetEvent* event);
    static bool eventIsImportant(Event event);
};
