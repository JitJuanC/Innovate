import asyncio
import websockets
import time as time_module
from datetime import datetime, time, timedelta
import MySQLdb
from email_sending import send_email, email_body
from excel_write_temi import create_workbook


def main():
    def get_stars(cursor_obj, sending_date, i):
        cursor_obj.execute(f"select count(stars) from feedback where time like '{sending_date}%' and stars like {i}")
        star = cursor_obj.fetchall()
        return star[0][0]

    async def timer():
        print("Starting timer to 5051")
        start = time_module.time() # unix time
        feedback_sent = False # Flag for sending email once
        while True:
            now = time_module.time()
            if now - start >= 30:
                async with websockets.connect('ws://127.0.0.1:5051') as ws:
                    await ws.send('go')
                start = now
            if time.min < datetime.now().time() < time(1) and feedback_sent == False:
                feedback_sent = True
                sending_date = str(datetime.now() - timedelta(hours=12)).split(' ')[0] # Yesterday's date
                #sending_date = "2021-10-19"
                try:
                    db = MySQLdb.connect('172.18.0.3', 'graymatics', 'graymatics', db='dbs') # Connect to the MySQL database
                    cursor = db.cursor()
                    # Sum of all feedbacks
                    cursor.execute(f"select cast(avg(f.stars) as decimal(10,1)), sum(p.intuitive), sum(p.volume), sum(p.comprehensive), \
                        sum(p.directing), sum(ft.video_call), sum(ft.show_location), sum(ft.unattended), sum(ft.transaction) \
                        from feedback f join performance p on f.id=p.id join features ft on f.id=ft.id where f.time like '{sending_date}%'")
                    fetched = cursor.fetchall() 
                    stars = {i : get_stars(cursor, sending_date, i) for i in range(1, 6)} 
                    feedback_sum = [int(i) for i in fetched[0][1:]]

                    # Email Body
                    text = email_body(f""" 
                    AUTOMATICALLY GENERATED EMAIL BY GRAYMATICS/TEMI AT 0000 HOURS

                    Date: {sending_date}

                    Average rating of stars: {fetched[0][0]}

                    Performance Count(s) Received Today:
                        The chatbot experience (e.g. intuitive) - {fetched[0][1]}
                        Clarity of the announcements (e.g. volume) - {fetched[0][2]}
                        Ease of understanding the FAQ page (e.g. comprehensive) - {fetched[0][3]}
                        Navigation feature (e.g. directing me to the appropriate machine) - {fetched[0][4]}

                    Features Count(s) Received Today:
                        Video call with staff for General Enquiry - {fetched[0][5]}
                        Show locations of nearby branches & ATMs - {fetched[0][6]}
                        Notify customers of their unattended items (e.g. Wallet, Bag) - {fetched[0][7]}
                        Step-by-step guide to perform a particular transaction (eg. Top-up of Ezlink card) - {fetched[0][8]}

                    """)
                    create_workbook(sending_date, stars, feedback_sum) 
                    # time_module.sleep(2)
                    send_email(["dbsgraymatics@gmail.com", "jovy@mirobotic.sg"], text, sending_date)
                    cursor.close()
                    db.close() # Close the MySQL connection
                except Exception as e:
                    print(e)
                    # pass
            if time(22) < datetime.now().time() < time(23):
                feedback_sent = False

                # select f.id, p.intuitive, p.volume, p.comprehensive, p.directing from feedback f join performance p on f.id=p.id where f.time like "2021-10-07%";

    asyncio.get_event_loop().run_until_complete(timer())


if __name__ == '__main__':
    # testing in production env
    main()
    # db = MySQLdb.connect('172.18.0.3', 'graymatics', 'graymatics', db='dbs')
    # cursor = db.cursor()
    # sending_date = "2021-10-07" # Yesterday's date
    # def get_stars(sending_date, i):
    #     cursor.execute(f"select count(stars) from feedback where time like '{sending_date}%' and stars like {i}")
    #     star = cursor.fetchall()
    #     return star[0][0]
    
    # # Sum of all feedbacks
    # cursor.execute(f"select cast(avg(f.stars) as decimal(10,1)), sum(p.intuitive), sum(p.volume), sum(p.comprehensive), \
    #     sum(p.directing), sum(ft.video_call), sum(ft.show_location), sum(ft.unattended), sum(ft.transaction) \
    #     from feedback f join performance p on f.id=p.id join features ft on f.id=ft.id where f.time like '{sending_date}%'")
    # fetched = cursor.fetchall() 
    # stars = {i : get_stars(sending_date, i) for i in range(1, 6)} 
    # feedback_sum = [int(i) for i in fetched[0][1:]]

    # # Email Body
    # text = email_body(f""" 
    # AUTOMATICALLY GENERATED EMAIL BY GRAYMATICS/TEMI AT 0000 HOURS

    # Date: {sending_date}

    # Average rating of stars: {fetched[0][0]}

    # Performance Count(s) Received Today:
    #     The chatbot experience (e.g. intuitive) - {fetched[0][1]}
    #     Clarity of the announcements (e.g. volume) - {fetched[0][2]}
    #     Ease of understanding the FAQ page (e.g. comprehensive) - {fetched[0][3]}
    #     Navigation feature (e.g. directing me to the appropriate machine) - {fetched[0][4]}

    # Features Count(s) Received Today:
    #     Video call with staff for General Enquiry - {fetched[0][5]}
    #     Show locations of nearby branches & ATMs - {fetched[0][6]}
    #     Notify customers of their unattended items (e.g. Wallet, Bag) - {fetched[0][7]}
    #     Step-by-step guide to perform a particular transaction (eg. Top-up of Ezlink card) - {fetched[0][8]}

    # """)
    # create_workbook(sending_date, stars, feedback_sum) 
    # # time_module.sleep(2)
    # send_email(["dbsgraymatics@gmail.com", "jovy@mirobotic.sg"], text, sending_date)
