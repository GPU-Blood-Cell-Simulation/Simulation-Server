#include "msg_receiver.hpp"

#include <stdexcept>


#define BITS_IN_BYTE 8
#define START_EVENT_MASK (eventType)(1 << (sizeof(eventType) * BITS_IN_BYTE - 1))
#define EVENT_PAYLOAD (~START_EVENT_MASK)


MsgReceiver::MsgReceiver(int port, const std::string &address)
{
    if (enet_initialize() != 0) {
        throw std::runtime_error("Cannot initialize message receiver");
    }

    /* Bind to local host */
    this->address.host = ENET_HOST_ANY;
    //enet_address_set_host_ip(&this->address, "127.0.0.1");
    this->address.port = port;

    server = enet_host_create(&this->address, 32, 2, 0, 0);
    
    if (server == NULL) {
        throw std::runtime_error("Cannot create a host");
    }
}

MsgReceiver::~MsgReceiver()
{
    enet_host_destroy(server);
    enet_deinitialize();
}


ReceivedEvent MsgReceiver::pollMessages() const
{
    ENetEvent event;
    ReceivedEvent resultEvent(Event::noMessage);

    if (enet_host_service(server, &event, 0) < 0) {
        throw std::runtime_error("Error while polling messages");
    }

    switch (event.type)
    {
    case ENET_EVENT_TYPE_NONE:
        return ReceivedEvent(Event::noMessage);

    case ENET_EVENT_TYPE_CONNECT:
        return ReceivedEvent(Event::newConnection);

    case ENET_EVENT_TYPE_DISCONNECT:
        return ReceivedEvent(Event::peerDisconnected);

    case ENET_EVENT_TYPE_RECEIVE:
        resultEvent = parseMessage(&event);
        enet_packet_destroy (event.packet);
        break;
    
    default:
        throw std::runtime_error("Undefined message type");
    }

    return resultEvent;
}

ReceivedEvent MsgReceiver::parseMessage(const ENetEvent *event) const
{
    if (event->packet->dataLength != sizeof(Event)) {
        return ReceivedEvent(Event::invalidMessage);
    }

    if (static_cast<eventType>(*event->packet->data) & EVENT_PAYLOAD >= static_cast<eventType>(Event::invalidMessage)) {
        return ReceivedEvent(Event::invalidMessage);
    }

    return ReceivedEvent(
        static_cast<Event>(*event->packet->data & EVENT_PAYLOAD),
        (*event->packet->data & START_EVENT_MASK) == 0 ? EventState::stop : EventState::start
    );
}

ReceivedEvent::ReceivedEvent(Event event, EventState state):
    event(event), state(state) {}
