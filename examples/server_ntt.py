import asyncio
import websockets

def main():
    port = 5050
    host = 'localhost'
    connected = set()
    ws1 = websockets.connect('ws://127.0.0.1:5051')
    print(f"Server started for COMPANY --> instance: {ws1}")

    # async def establish():
    #     async with websockets.connect('ws://127.0.0.1:5051') as ws:
    #         await ws.send('Establishing connection')
    #         return ws
    # websockets.serve

    # connected.add(asyncio.open_connection('ws://127.0.0.1:5051'))

    async def handle(websocket, path):
        connected.add(websocket)
        print(connected)
        try:
            async for data in websocket:
                print("Received data: " + data)
                for client in connected:
                    try:
                        await client.send(data)
                        # websockets.broadcast(connected, data)
                    except AttributeError:
                        continue
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Client {websocket} is disconnected")
        finally:
            connected.remove(websocket)

    start = websockets.serve(handle, host, port)
    asyncio.get_event_loop().run_until_complete(start)
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    main()
