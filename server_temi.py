import MySQLdb
from datetime import datetime
import asyncio
import websockets
import json
import uuid
import time
from use.alert import send_ws

def main():
    # if not exists, initialize the mysql table
    mysql_args = {
        "ip":'172.18.0.3',
        "user":'graymatics',
        "pwd":'graymatics',
        "db":"dbs",

        "table_feedback":"feedback",
        "column_feedback":[['id','varchar(45)'],
            ['time','datetime'],
            ['stars','int(10)']],

        "table_alert":"alert",
        "column_alert":[['id','varchar(45)'],
            ['time','datetime'],
            ['mgs','varchar(45)']]
        }
    # temi's data names
    data_names = ('feedback', 'alert')
    db = MySQLdb.connect(mysql_args['ip'], mysql_args['user'], mysql_args['pwd'], db=mysql_args['db'])
    cursor = db.cursor()
    for data_name in data_names:
        column_initialize = ','.join([f'{c[0]} {c[1]}' for c in mysql_args[f'column_{data_name}']])
        cursor.execute(f"create table if not exists {mysql_args[f'table_{data_name}']} ({column_initialize})")
    # feedback additional tables ---------------------
    cursor.execute(f"create table if not exists performance (id varchar(45), intuitive int(1), volume int(1), \
        comprehensive int(1), directing int(1))") # create separate table to have a more detailed table, later inner join with id
    cursor.execute(f"create table if not exists features (id varchar(45), video_call int(1), show_location int(1), \
        unattended int(1), transaction int(1))")
    # ------------------------------------------------
    db.commit() 
    cursor.close()
    db.close()

    def add_feedback(ws_data): # dumped json (string)
        try:
            db = MySQLdb.connect(mysql_args['ip'], mysql_args['user'], mysql_args['pwd'], db=mysql_args['db']) # Refresh the MySQL Connection
            cursor = db.cursor()
            ws_data = json.loads(ws_data) # convert str/bytes to json
            uuid_temi = str(uuid.uuid4())
            now_time = datetime.now()
            requested = ws_data.get('request').lower()

            # take action when there is data from temi which is in data_names
            if requested in data_names:
                column = ','.join([f'{c[0]}' for c in mysql_args[f'column_{requested}']])
                if requested == 'feedback':
                    performance = str(ws_data.get('performance')) # Get Performance
                    intuitive = 0 if performance.find('intuitive') == -1 else 1
                    volume = 0 if performance.find('volume') == -1 else 1
                    comprehensive = 0 if performance.find('comprehensive') == -1 else 1
                    directing = 0 if performance.find('directing') == -1 else 1

                    features = str(ws_data.get('features')) # Get Features
                    video_call = 0 if features.find('call') == -1 else 1
                    show_location = 0 if features.find('locations') == -1 else 1
                    unattended = 0 if features.find('unattended') == -1 else 1
                    transaction = 0 if features.find('transaction') == -1 else 1

                    cmd = f"insert into {mysql_args[f'table_{requested}']} ({column}) values (\"{uuid_temi}\", \"{now_time}\", \
                        {int(ws_data.get('stars'))})"
                    cursor.execute(cmd)
                    cursor.execute(f"insert into performance (id, intuitive, volume, comprehensive, directing) values (\"{uuid_temi}\", \
                        {intuitive}, {volume}, {comprehensive}, {directing})")
                    cursor.execute(f"insert into features (id, video_call, show_location, unattended, transaction) values (\"{uuid_temi}\", \
                        {video_call}, {show_location}, {unattended}, {transaction})")
                    db.commit()
                if requested == 'alert':
                    data = {'data':{'time':time.time(), 'cam_name':'Temi_Robot', 'cam_id':'3', 'alert':'temi lift alert'}, 'algo':'temi'}
                    send_ws('ntt', data)
                    cmd = f"insert into {mysql_args[f'table_{requested}']} ({column}) values (\"{uuid_temi}\", \"{now_time}\", \
                        \"{str(ws_data.get('mgs'))}\")"
                    cursor.execute(cmd)
                    db.commit()
            cursor.close() # Close the MySQL connection
            db.close()
        except Exception as e:
            print(e)
            print('Error Writing to MySQL')

    port = 5052 # Port for clients (sender) to connect to this server
    host = 'localhost'
    connected = set()
    alarms = set()
    print(f"Server started at 5051")  # This port is for the client (receiver) to connect to this server

    async def send(websocket, path):
        connected.add(websocket) # Add client websockets
        try:
            async for data in websocket:
                if data.lower() == 'go':
                    connected.remove(websocket) # if client is the timer, remove that websocket to prevent clogging the client (receiver)
                    if len(alarms) > 0:
                        if len(connected) > 0:
                            if 'mask' in alarms:
                                sending = {"alert": "mask", "time": time.time()} # json format
                                sending = json.dumps(sending)
                                for client in connected: # send to all connected clients (receiver)
                                    await client.send(sending) # asynchronus to make sure it sends to 1 client at a time, will not proceed if stuck
                                    # print('Sent', sending)
                            await asyncio.sleep(2) # asynchronus sleep function to stagger the sending function every 2 seconds without affecting other functions, client only can receive the data with 2 seconds cooldown period
                            if 'socialD' in alarms:
                                sending = {"alert": "socialD", "time": time.time()}
                                sending = json.dumps(sending)
                                for client in connected:
                                    await client.send(sending)
                                    # print('Sent', sending)
                            await asyncio.sleep(2)
                            if 'occupancyCount' in alarms:
                                sending = {"alert": "occupancyCount", "time": time.time()}
                                sending = json.dumps(sending)
                                for client in connected:
                                    await client.send(sending)
                                    # print('Sent', sending)
                        alarms.clear() # reset the alarms  
                else: 
                    print("DATA FROM TEMI: ", data)
                    print(type(data))
                    add_feedback(data)
        except websockets.exceptions.ConnectionClosed as e:
            # pass
            print(f"Client {websocket} is disconnected") # client (receiver) disconnect
        finally:
            if websocket in connected: # make sure the client (receiver) that has just disconnected is in the set of clients that are waiting to receive
                connected.remove(websocket) 

    async def handle(websocket, path): # different asynchronus function from send()
        try:
            async for data in websocket:
                print("Received data from graymatics: " + data) # prints the data from client (sender)
                converted_data = json.loads(data) # convert to dictionary format (json)
                if converted_data.get('alert') not in alarms: # check the alert from the dictionary, if have - add to the alarm set
                    alarms.add(converted_data.get('alert'))
        except websockets.exceptions.ConnectionClosed as e:
            pass # catch exception when client (sender) has sent data and closed connection

    # serve the websocket
    start = websockets.serve(handle, host, port) # server for client (sender)
    sender = websockets.serve(send, host, 5051) # server for client (receive) / broadcasting server
    asyncio.get_event_loop().run_until_complete(start)
    asyncio.get_event_loop().run_until_complete(sender)
    asyncio.get_event_loop().run_forever() # run the servers asynchronusly so other non-asynchronus parts of the script will not be affected

if __name__ == '__main__':
    main()
