#include "server_endpoint.hpp"

#include <stdexcept>


#define BITS_IN_BYTE 8
#define START_EVENT_MASK (event_t)(1 << (sizeof(event_t) * BITS_IN_BYTE - 1))
#define EVENT_PAYLOAD (~START_EVENT_MASK)

#define NUMBER_OF_CHANNELS 2
#define STD_CHANNEL_ID 0
#define IMPORTANT_CHANNEL_ID 1


int ServerCommunicationEndpoint::instances = 0;


ServerCommunicationEndpoint::ServerCommunicationEndpoint(int port, const std::string &address)
{
    if (instances++ == 0) {
        if (enet_initialize() != 0) {
            throw std::runtime_error("Cannot initialize message receiver");
        }
    }

    /* Bind to local host */
    if (address == "localhost") {
        this->address.host = ENET_HOST_ANY;
    }
    else if (enet_address_set_host(&this->address, address.c_str())) {
        throw std::runtime_error("Error while setting the host");
    }
    
    this->address.port = port;

    server = enet_host_create(
        &this->address,
        1,                  // Max number of clients
        NUMBER_OF_CHANNELS,
        0,                  // Accept any amount of incoming bandwidth
        0                   // Accept any amount of outgoing bandwidth
    );
    
    if (server == NULL) {
        throw std::runtime_error("Cannot create a host");
    }
}

ServerCommunicationEndpoint::~ServerCommunicationEndpoint()
{
    enet_host_destroy(server);

    if (--instances == 0) {
        enet_deinitialize();
    }
}


Event ServerCommunicationEndpoint::pollEvents()
{
    ENetEvent event;
    Event resultEvent(EventType::noMessage);

    if (enet_host_service(server, &event, 0) < 0) {
        throw std::runtime_error("Error while polling messages");
    }

    switch (event.type)
    {
    case ENET_EVENT_TYPE_NONE:
        return Event(EventType::noMessage);

    case ENET_EVENT_TYPE_CONNECT:
        peer = event.peer;
        return Event(EventType::newConnection);

    case ENET_EVENT_TYPE_DISCONNECT:
        return Event(EventType::peerDisconnected);

    case ENET_EVENT_TYPE_RECEIVE:
        resultEvent = parseEvent(&event);
        enet_packet_destroy (event.packet);
        break;
    
    default:
        throw std::runtime_error("Undefined message type");
    }

    return resultEvent;
}


bool ServerCommunicationEndpoint::isConnected() const
{
    if (peer != NULL && peer->state == ENET_PEER_STATE_CONNECTED) {
        return true;
    }

    return false;
}


void ServerCommunicationEndpoint::SendEvent(Event event) const
{
    if (!isConnected()) {
        return;
    }

    event_t eventMsg = static_cast<event_t>(event.eventType);

    if (event.state == EventState::start) {
        eventMsg |= START_EVENT_MASK;
    }

    ENetPacket* packet = enet_packet_create(&eventMsg, sizeof(event_t), ENET_PACKET_FLAG_RELIABLE);
    if (packet == NULL) {
        throw std::runtime_error("Error while creating a packet");
    }

    enet_uint8 channel = eventIsImportant(event) ? IMPORTANT_CHANNEL_ID : STD_CHANNEL_ID;

    if (enet_peer_send(peer, channel, packet) != 0) {
        throw std::runtime_error("Error while sending a packet");
    }
}


Event ServerCommunicationEndpoint::parseEvent(const ENetEvent *event)
{
    static_assert((event_t)EventType::invalidMessage < START_EVENT_MASK);

    if (event->packet->dataLength != sizeof(EventType)) {
        return Event(EventType::invalidMessage);
    }

    event_t eventPayload = static_cast<event_t>(*event->packet->data) & static_cast<event_t>(EVENT_PAYLOAD);
    if (eventPayload >= static_cast<event_t>(EventType::invalidMessage)) {
        return Event(EventType::invalidMessage);
    }

    return Event(
        static_cast<EventType>(*event->packet->data & EVENT_PAYLOAD),
        (*event->packet->data & START_EVENT_MASK) == 0 ? EventState::stop : EventState::start
    );
}

bool ServerCommunicationEndpoint::eventIsImportant(Event event)
{
    switch (event.eventType)
    {
    case EventType::streamSuccessfullyEnded:
    case EventType::stopRendering:
        return true;
    
    default:
        return false;
    }
}

Event::Event(EventType event, EventState state):
    eventType(event), state(state) {}
