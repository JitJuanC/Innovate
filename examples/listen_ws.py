import websockets
import asyncio

async def listen():
    url = "ws://0.tcp.ngrok.io:12242" # target this server to listen in 
    print(f"Successfully connected to this websocket server: {url}")
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv() # once successfully connected, just wait for the server to broadcast a string, then wait again (without affecting others in script because async) because of await
            print(msg)

asyncio.get_event_loop().run_until_complete(listen()) # forever, because while True
# asyncio.get_event_loop().run_forever() 
# asyncio.run(listen())